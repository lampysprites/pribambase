# Copyright (c) 2021 lampysprites
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import bpy
import imbuf
import os
from os import path
import tempfile
import numpy as np
from typing import Collection, Tuple

from .addon import addon


def refresh():
    """Tag the ui for redrawing"""
    ctx = bpy.context
    if not ctx or not ctx.window_manager:
        return
    
    for win in ctx.window_manager.windows:
        for area in win.screen.areas:
            area.tag_redraw()


class ModalExecuteMixin:
    """
    bpy.types.Operator mixin that makes operator execute once via modal timer, allowing to modify 
    blender state from non-operator code with fewer surprizes. Uses a non-modal fallback for older
    versions. To use, define modal_execute(self, ctx) method
    """

    def modal_execute(self, context):
        raise NotImplementedError()

    def modal(self, context, event):
        if event.type == 'TIMER':
            context.window_manager.event_timer_remove(self.timer)
            self.modal_execute(context)
        return {'FINISHED'}

    def execute(self, context):
        if context and context.window and not addon.prefs.skip_modal:
            context.window_manager.modal_handler_add(self)
            self.timer = context.window_manager.event_timer_add(0.000001, window=context.window)
            return {'RUNNING_MODAL'}
        else:
            return self.modal_execute(context)


def image_name(img):
    fp = img.filepath
    name = img.name

    if img.sb_props.source:
        name = os.path.normpath(img.sb_props.source_abs)

    elif not img.packed_file and fp:
        name = os.path.normpath(bpy.path.abspath(fp) if fp.startswith("//") else fp)

    return name


def new_packed_image(name, w, h):
    """Create a packed image with data that will be saved (unlike bpy.data.images.new that is cleaned when the file is opened)"""
    img = bpy.data.images.new(name, w, h, alpha=True)
    tmp = path.join(tempfile.gettempdir(), "__sb__delete_me.png")
    img.filepath = tmp
    img.save() # the file needs to exist for pack() to work
    img.pack()
    img.filepath=""
    img.use_fake_user = True
    os.remove(tmp)
    return img


_update_image_args = None
def update_image(w, h, name, frame, pixels):
    global _update_image_args
    _update_image_args = w, h, name, frame, pixels
    bpy.ops.pribambase.update_image()

class SB_OT_update_image(bpy.types.Operator, ModalExecuteMixin):
    bl_idname = "pribambase.update_image"
    bl_label = "Update Image"
    bl_description = ""
    bl_options = {'REGISTER', 'UNDO_GROUPED', 'INTERNAL'}
    bl_undo_group = "pribambase.update_image"

    def modal_execute(self, context):
        """Replace the image with pixel data"""
        img = None
        w, h, name, frame, pixels = self.args

        try:
            img = next(i for i in bpy.data.images if name == image_name(i))
        except StopIteration:
            # to avoid accidentally reviving deleted images, we ignore anything doesn't exist already
            return

        if not img.has_data:
            # load *some* data so that the image can be packed, and then updated
            ib = imbuf.new((w, h))
            tmp = path.join(tempfile.gettempdir(), "__sb__delete_me.png")
            imbuf.write(ib, tmp)
            img.filepath = tmp
            img.reload()
            img.pack()
            img.filepath=""
            img.use_fake_user = True
            os.remove(tmp)

        elif (img.size[0] != w or img.size[1] != h):
                img.scale(w, h)
        
        if frame != -1:
            img.sb_props.frame = frame

        # convert data to blender accepted floats
        pixels = np.float32(pixels) / 255.0
        # flip y axis ass backwards
        pixels.shape = (h, pixels.size // h)
        pixels = pixels[::-1,:].ravel()

        # change blender data
        try:
            # version >= 2.83; this is much faster
            img.pixels.foreach_set(pixels)
        except AttributeError:
            # version < 2.83
            img.pixels[:] = pixels

        img.update()

        # [#12] for some users viewports do not update from update() alone
        img.update_tag()
        refresh()
        
        self.args = None
        global _update_image_args
        _update_image_args = None

        return {'FINISHED'}


    def execute(self, context):
        self.args = _update_image_args
        return ModalExecuteMixin.execute(self, context)


_update_spritesheet_args = None
def update_spritesheet(size, count, name, start, frames, tags, current_frame, current_tag, pixels):
    global _update_spritesheet_args
    _update_spritesheet_args = size, count, name, start, frames, tags, current_frame, current_tag, pixels
    bpy.ops.pribambase.update_spritesheet()
    
class SB_OT_update_spritesheet(bpy.types.Operator, ModalExecuteMixin):
    bl_idname = "pribambase.update_spritesheet"
    bl_label = "Update Spritesheet"
    bl_description = ""
    bl_options = {'REGISTER', 'UNDO_GROUPED', 'INTERNAL'}
    bl_undo_group = "pribambase.update_spritesheet"


    def update_actions(self, context, img_name:str, start:int, frames:Collection[int], tags:Collection[Tuple[str, int, int, int]], current_tag:str):
        fps = context.scene.render.fps / context.scene.render.fps_base

        # editor tag is the current play loop in aseprite
        tag_editor = ("",)
        if current_tag:
            tag_editor += next((t for t in tags if t[0] == current_tag))[1:]
        else:
            tag_editor += (start, start + len(frames), 0)

        for tag, tag_first, tag_last, ani_dir in (tag_editor, *tags):
            action_name = img_name if tag == "" else f"{img_name}: {tag}"
            try:
                action = bpy.data.actions[action_name]
            except KeyError:
                action = bpy.data.actions.new(action_name)
                action.id_root = 'OBJECT'
                action.use_fake_user = True
            
            fcurve = action.fcurves.find('["Sprite Frame"]')
            if not fcurve:
                fcurve = action.fcurves.new('["Sprite Frame"]')
            
            time = 0
            first = context.scene.frame_start

            tag_frames = frames[tag_first - start:tag_last - start + 1]
            if ani_dir == 1:
                tag_frames = tag_frames[::-1]
            elif ani_dir == 2:
                tag_frames = tag_frames + tag_frames[-2:0:-1] # sigh
            
            tag_frames.append(tag_frames[-1]) # one more keyframe to keep the last frame duration inside in the action

            points = fcurve.keyframe_points
            npoints = len(points)
            nframes = len(tag_frames)
            if npoints < nframes:
                points.add(nframes - npoints)
            elif npoints > nframes:
                for _ in range(npoints - nframes):
                    points.remove(points[0], fast=True)

            for point,(n, dt) in zip(points, tag_frames):
                point.co = (first + time * fps / 1000, n)
                point.select_control_point = point.select_left_handle = point.select_right_handle = False
                point.interpolation = 'CONSTANT'
                time += dt

            fcurve.update()
            action.update_tag()


    def modal_execute(self, context):
        size, count, name, start, frames, tags, current_frame, current_tag, pixels = self.args
        tex_w, tex_h = size[0] * count[0], size[1] * count[1]

        # find or prepare sheet image; pixels update will fix its size
        try:
            img = next(i for i in bpy.data.images if name == image_name(i))
        except StopIteration:
            # did not set up the texture first, or deleted it
            return
        
        try:
            sheet = img.sb_props.sheet
            tex_name = sheet.name
        except AttributeError:
            tex_name = img.name + " [sheet]"
            if tex_name not in bpy.data.images:
                new_packed_image(tex_name, tex_w, tex_h)
            sheet = img.sb_props.sheet = bpy.data.images[tex_name]
        
        sheet.sb_props.is_sheet = True
        sheet.sb_props.sheet_size = count
        sheet.sb_props.sheet_start = start

        # fill the frame data; first make sure there's enough and not too many frames
        fd = sheet.sb_props.sheet_frames
        for _ in range(len(fd) - len(frames)):
            fd.remove(0)
        for _ in range(len(frames) - len(fd)):
            fd.add()

        t = 0
        for i,(cel, dt) in enumerate(frames):
            f = fd[i]
            f.frame = i + start
            f.time = t / 1000.0
            f.index = i # TODO cel optimization
            t += dt

        self.update_actions(context, img.name, start, frames, tags, current_tag)

        self.args = tex_w, tex_h, tex_name, -1, pixels
        SB_OT_update_image.modal_execute(self, context) # clears self.args

        # cut out the current frame and copy to view image
        frame_x = current_frame % count[0]
        frame_y = current_frame // count[0]
        frame_pixels = np.ravel(pixels[frame_y * size[1] : (frame_y + 1) * size[1], frame_x * size[0] * 4 : (frame_x + 1) * size[0] * 4])
        self.args = *size, name, current_frame, frame_pixels
        SB_OT_update_image.modal_execute(self, context) # clears self.args

        # clean up
        global _update_spritesheet_args
        _update_spritesheet_args = None
    
    def execute(self, context):
        self.args = _update_spritesheet_args
        return ModalExecuteMixin.execute(self, context)


class SB_OT_report(bpy.types.Operator, ModalExecuteMixin):
    bl_idname = "pribambase.report"
    bl_label = "Report"
    bl_description = "Report the message"
    bl_options = {'INTERNAL'}

    message_type: bpy.props.StringProperty(name="Message Type", default='INFO')
    message: bpy.props.StringProperty(name="Message", default='Someone forgot to change the message text')

    def modal_execute(self, context):
        self.report({self.message_type}, self.message)
        return {'FINISHED'}