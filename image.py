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
from bpy.app.translations import pgettext
import bmesh
import gpu
import bgl
from mathutils import Matrix
from gpu_extras.batch import batch_for_shader
import numpy as np
from os import path

from .messaging import encode
from . import util
from .addon import addon


COLOR_MODES = [
    ('rgba', "RGBA", "32-bit color with transparency. If not sure, pick this one"),
    ('indexed', "Indexed", "Palettized image with arbitrary palette"),
    ('gray', "Grayscale", "Palettized with 256 levels of gray")]

UV_DEST = [
    ('texture', "Texture Source", "Show UV map in the file of the image editor's texture"),
    ('active', "Active Sprite", "Show UV map in the currently open document")
]


class SB_OT_uv_send(bpy.types.Operator):
    bl_idname = "pribambase.uv_send"
    bl_label = "Send UV"
    bl_description = "Show UV in Aseprite"


    destination: bpy.props.EnumProperty(
        name="Show In",
        description="Which document's UV map will be shown in aseprite",
        items=UV_DEST,
        default='texture')


    size: bpy.props.IntVectorProperty(
        name="Resolution",
        description="The size for the created UVMap. The image is scaled to the size of the sprite",
        size=2,
        min=1,
        max=65535,
        default=(1, 1))

    color: bpy.props.FloatVectorProperty(
        name="Color",
        description="Color to draw the UVs with",
        size=4,
        min=0.0,
        max=1.0,
        default=(0.0, 0.0, 0.0, 0.0),
        subtype='COLOR')

    weight: bpy.props.FloatProperty(
        name="Thickness",
        description="Thickness of the UV map lines at its original resolution",
        min=0,
        max=65535,
        default=0)


    @classmethod
    def poll(self, context):
        return addon.connected and context.edit_object is not None or context.image_paint_object is not None


    def list_uv(self):
        ctx = bpy.context
        active = ctx.object
        lines = set()

        objects = [obj for obj in ctx.selected_objects if obj.type == 'MESH']
        if (active is not None) and (active not in objects) and (active.type == 'MESH'):
            objects.append(ctx.object)

        for obj in objects:

            try:
                bm = bmesh.from_edit_mesh(obj.data)
                bm_created = False # freeing an editmode bmesh crashes blender
            except: # if there's `elif`, why isn't there `exceptry`?
                try:
                    bm = bmesh.new()
                    bm_created = True
                    bm.from_mesh(obj.data)
                except:
                    self.report('WARNING', "UVMap drawing skipped: can't access mesh data")
                    continue

            uv = bm.loops.layers.uv.active

            # get all edges
            for face in bm.faces:
                if not face.select:
                    continue

                for i in range(0, len(face.loops)):
                    a = face.loops[i - 1][uv].uv.to_tuple()
                    b = face.loops[i][uv].uv.to_tuple()

                    # sorting prevents the edge from being added twice for differently directed loops
                    # order doesn't really matter, just that there is one
                    if a > b:
                        a, b = b, a

                    lines.add((a, b))

            if bm_created:
                bm.free()

        return lines


    def uvmap_size(self):
        scale = addon.prefs.uv_scale
        size = [128, 128]

        if bpy.context.edit_image is not None:
            size = bpy.context.edit_image.size

        return [int(size[0] * scale), int(size[1] * scale)]


    def execute(self, context):
        w, h = self.size
        source = ""

        if self.destination == 'texture':
            source = context.area.spaces.active.image.sb_props.sync_name

        aa = addon.prefs.uv_aa
        weight = self.weight
        lines = self.color[0:3] + (1.0,)
        nbuf = np.zeros((h, w, 4), dtype=np.uint8)

        offscreen = gpu.types.GPUOffScreen(w, h)

        coords = [c for pt in self.list_uv() for c in pt]
        shader = gpu.shader.from_builtin('2D_UNIFORM_COLOR')
        batch = batch_for_shader(shader, 'LINES', {"pos": coords})

        with offscreen.bind():
            with gpu.matrix.push_pop():
                # see explanation in https://blender.stackexchange.com/questions/153697/gpu-python-module-why-drawed-pixels-are-shifted-in-the-result-image
                projection_matrix = Matrix.Diagonal((2.0, -2.0, 1.0))
                projection_matrix = Matrix.Translation((-1.0 + 1.0 / w, 1.0 + 1.0 / h, 0.0)) @ projection_matrix.to_4x4()
                gpu.matrix.load_projection_matrix(projection_matrix)

                bgl.glEnable(bgl.GL_BLEND)
                bgl.glBlendFunc(bgl.GL_SRC_ALPHA, bgl.GL_ONE_MINUS_SRC_ALPHA)
                bgl.glLineWidth(weight)

                if aa:
                    bgl.glEnable(bgl.GL_BLEND)
                    bgl.glBlendFunc(bgl.GL_SRC_ALPHA, bgl.GL_ONE_MINUS_SRC_ALPHA)
                    bgl.glEnable(bgl.GL_LINE_SMOOTH)
                    bgl.glHint(bgl.GL_LINE_SMOOTH_HINT, bgl.GL_NICEST)
                else:
                    bgl.glDisable(bgl.GL_BLEND)
                    bgl.glDisable(bgl.GL_LINE_SMOOTH)
                    bgl.glHint(bgl.GL_LINE_SMOOTH_HINT, bgl.GL_FASTEST)

                shader.bind()
                shader.uniform_float("color", lines)
                batch.draw(shader)

            # retrieve the texture
            # https://blender.stackexchange.com/questions/221110/fastest-way-copying-from-bgl-buffer-to-numpy-array
            buffer = bgl.Buffer(bgl.GL_BYTE, nbuf.shape, nbuf)
            bgl.glReadPixels(0, 0, w, h, bgl.GL_RGBA, bgl.GL_UNSIGNED_BYTE, buffer)

        # send data
        msg = encode.uv_map(
                size=(w, h),
                sprite=source,
                pixels=nbuf.tobytes(),
                layer=addon.prefs.uv_layer,
                opacity=int(addon.prefs.uv_color[3] * 255))
        if source:
            msg = encode.batch((encode.sprite_focus(source), msg))

        addon.server.send(msg)

        return {"FINISHED"}


    def invoke(self, context, event):
        if tuple(self.size) == (1, 1):
            self.size = self.uvmap_size()

        if tuple(self.color) == (0.0, 0.0, 0.0, 0.0):
            self.color = addon.prefs.uv_color

        if self.weight == 0.0:
            self.weight = addon.prefs.uv_weight

        return context.window_manager.invoke_props_dialog(self)



class SB_OT_sprite_open(bpy.types.Operator):
    bl_idname = "pribambase.sprite_open"
    bl_label = "Open..."
    bl_description = "Set up a texture from a file using Aseprite"
    bl_options = {'UNDO'}


    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    relative: bpy.props.BoolProperty(name="Relative Path", description="Select the file relative to blend file")
    sheet: bpy.props.BoolProperty(name="Animation", description="If checked, entire animation will be synced to blender; if not, only the current frame will. Same as 'Animation' switch in Aseprite's sync popup")

    # dialog settings
    filter_glob: bpy.props.StringProperty(default="*.ase;*.aseprite;*.bmp;*.flc;*.fli;*.gif;*.ico;*.jpeg;*.jpg;*.pcx;*.pcc;*.png;*.tga;*.webp", options={'HIDDEN'})
    use_filter: bpy.props.BoolProperty(default=True, options={'HIDDEN'})


    @classmethod
    def poll(self, context):
        return addon.connected


    def execute(self, context):
        _, name = path.split(self.filepath)
        img = None

        self.__class__._last_relative = self.relative

        try:
            # we might have this image opened already
            img = next(i for i in bpy.data.images if i.sb_props.source_abs == self.filepath)
        except StopIteration:
            # create a stub that will be filled after receiving data
            with util.pause_depsgraph_updates():
                img = bpy.data.images.new(name, 1, 1, alpha=True)
                util.pack_empty_png(img)
                img.sb_props.source_set(self.filepath, self.relative)

        # switch to the image in the editor
        if context and context.area and context.area.type == 'IMAGE_EDITOR':
            context.area.spaces.active.image = img

        if addon.connected:
            msg = encode.sprite_open(name=self.filepath, flags={'SHEET'} if self.sheet else set())
            addon.server.send(msg)
        else:
            self.report({'WARNING'}, "Aseprite not connected - image data might not be loaded")

        return {'FINISHED'}


    def invoke(self, context, event):
        self.invoke_context = context

        # I have a feeling blender already has a solution but can't seem to find it
        if not hasattr(self.__class__, "_last_relative"):
            self.__class__._last_relative = addon.prefs.use_relative_path
        self.relative = self.__class__._last_relative

        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class SB_OT_sprite_new(bpy.types.Operator):
    bl_idname = "pribambase.sprite_new"
    bl_label = "New"
    bl_description = "Set up a new texture using Aseprite"
    bl_options={'UNDO'}

    sprite: bpy.props.StringProperty(
        name="Name",
        description="Name of the texture. It will also be displayed on the tab in Aseprite until you save the file",
        default="Sprite")

    size: bpy.props.IntVectorProperty(
        name="Size",
        description="Size of the created canvas",
        default=(128, 128),
        size=2,
        min=1,
        max=65535)

    mode: bpy.props.EnumProperty(
        name="Color Mode",
        description="Color mode of the created sprite",
        items=COLOR_MODES,
        default='rgba')


    @classmethod
    def poll(self, context):
        return addon.connected


    def execute(self, context):
        if not self.sprite:
            self.report({'ERROR'}, "The sprite must have a name")
            return {'CANCELLED'}

        # create a stub that will be filled after receiving data
        with util.pause_depsgraph_updates():
            img = bpy.data.images.new(self.sprite, 1, 1, alpha=True)
            util.pack_empty_png(img)
        # switch to it in the editor
        if context.area.type == 'IMAGE_EDITOR':
            context.area.spaces.active.image = img

        mode = 0
        for i,m in enumerate(COLOR_MODES):
            if m[0] == self.mode:
                mode = i

        msg = encode.sprite_new(
            name=img.name,
            size=self.size,
            flags=set(),
            mode=mode)

        addon.server.send(msg)

        return {'FINISHED'}


    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


class SB_OT_sprite_edit(bpy.types.Operator):
    bl_idname = "pribambase.sprite_edit"
    bl_label = "Edit"
    bl_description = "Open the file for this texture with Aseprite"


    @classmethod
    def poll(self, context):
        return addon.connected and context.area.type == 'IMAGE_EDITOR' \
            and context.edit_image and context.edit_image.has_data


    def execute(self, context):
        img = context.edit_image
        if img.sb_props.is_sheet:
            img = next((i for i in bpy.data.images if i.sb_props.sheet == img), img)
        edit_name = img.sb_props.sync_name
        msg = None

        if path.exists(edit_name):
            msg = encode.sprite_open(name=edit_name, flags=img.sb_props.sync_flags)
        else:
            pre_w = img.sb_props.prescale_size[0]
            desample = max(img.size[0] // pre_w, 1) if pre_w > 0 else 1
            pixels = np.asarray(np.array(img.pixels) * 255, dtype=np.ubyte)
            pixels.shape = (img.size[1], img.size[0], 4)
            pixels = np.ravel(pixels[::-desample,::desample,:])

            msg = encode.image(
                name=img.name,
                size=img.sb_props.prescale_size if pre_w > 0 else img.size,
                pixels=pixels.tobytes())

        addon.server.send(msg)

        return {'FINISHED'}


class SB_OT_sprite_edit_copy(bpy.types.Operator):
    bl_idname = "pribambase.sprite_edit_copy"
    bl_label = "Edit Copy"
    bl_description = "Open copy of the image in a new file in Aseprite, without syncing"


    @classmethod
    def poll(self, context):
        return addon.connected and context.area.type == 'IMAGE_EDITOR' \
            and context.edit_image and context.edit_image.has_data


    def execute(self, context):
        img = context.edit_image
        pre_w = img.sb_props.prescale_size[0]
        desample = max(img.size[0] // pre_w, 1) if pre_w > 0 else 1

        pixels = np.asarray(np.array(img.pixels) * 255, dtype=np.ubyte)
        pixels.shape = (img.size[1], img.size[0], 4)
        pixels = np.ravel(pixels[::-desample,::desample,:])

        msg = encode.image(
            name="",
            size=img.sb_props.prescale_size if pre_w > 0 else img.size,
            pixels=pixels.tobytes())

        addon.server.send(msg)

        return {'FINISHED'}


class SB_OT_sprite_purge(bpy.types.Operator):
    bl_label = "Purge"
    bl_idname = "pribambase.sprite_purge"
    bl_description = "Erase sprite-related data created by the plugin. Can remove unwanted animation setups"
    bl_options = {'UNDO'}


    remove_anim: bpy.props.BoolProperty(
        name="Animations", 
        description="Remove animations that use the sprite (UVWarp modifier, object property, and drivers)", 
        default=True)

    remove_actions: bpy.props.BoolProperty(
        name="Actions", 
        description="Remove generated actions and keyframes for the sprite", 
        default=True)

    remove_sheet: bpy.props.BoolProperty(
        name="Spritesheet Image", 
        description="Remove spritesheet image",
         default=True)

    remove_sprite: bpy.props.BoolProperty(
        name="Sprite Image", 
        description="Remove 'view' image. All relations to other pieces of data will be erased, so unchecking those makes IMPOSSIBLE to remove them automatically another time", 
        default=True)
    

    @classmethod
    def poll(cls, context):
        if not hasattr(context, "edit_image"): # FIXME do properly
            return False
        props = context.edit_image.sb_props
        return props.is_sheet or props.sheet
    
    def draw(self, context):
        row=self.layout.split(factor=.28)
        row.label(text="Remove:")
        col = row.column(align=True)
        col.prop(self, "remove_sprite")
        col.prop(self, "remove_sheet")
        col.prop(self, "remove_anim")
        col.prop(self, "remove_actions")
    

    def execute(self, context):
        if not self.img:
            self.report({'INFO'}, "The main sprite has been already removed, along with recorded relations. Some data items may require manual removal")
        
        if self.remove_anim:
            for obj in bpy.data.objects:
                for anim in obj.sb_props.animations:
                    if anim.image == self.img:
                        prop_name = anim.prop_name

                        # modifier
                        if obj.animation_data:
                            for driver in obj.animation_data.drivers:
                                if driver.data_path == f'modifiers["{prop_name}"].offset':
                                    obj.animation_data.drivers.remove(driver)

                        if prop_name in obj.modifiers:
                            obj.modifiers.remove(obj.modifiers[prop_name])

                        # custom property
                        try:
                            # 3.0+
                            obj.id_properties_ui(prop_name).clear()
                        except AttributeError:
                            # 2.[8/9]x
                            if "_RNA_UI" in obj and prop_name in obj["_RNA_UI"]:
                                del obj["_RNA_UI"][prop_name]

                        if prop_name in obj:
                            del obj[prop_name]

                        obj.sb_props.animations_remove(anim)


        if self.remove_actions:
            for action in bpy.data.actions:
                if action.sb_props.sprite == self.img:
                    bpy.data.actions.remove(action)
        
        if self.remove_sheet and self.sheet:
                bpy.data.images.remove(self.sheet)

        if self.remove_sprite and self.img:
                bpy.data.images.remove(self.img)

        return {'FINISHED'}


    def invoke(self, context, event):
        self.img = context.edit_image
        self.sheet = self.img.sb_props.sheet
        if self.img.sb_props.is_sheet: # user selected the spritesheet, not the "original" image
            sheet = self.img
            self.img = next((i for i in bpy.data.images if i.sb_props.sheet == sheet), None)
            self.remove_sprite = False # change default in this case

        return context.window_manager.invoke_props_dialog(self)


class SB_OT_sprite_replace(bpy.types.Operator):
    bl_description = "Replace current texture with a file using Aseprite"
    bl_idname = "pribambase.sprite_replace"
    bl_label = "Replace..."
    bl_options = {'UNDO'}

    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    relative: bpy.props.BoolProperty(name="Relative Path", description="Select the file relative to blend file")

    # dialog settings
    filter_glob: bpy.props.StringProperty(default="*.ase;*.aseprite;*.bmp;*.flc;*.fli;*.gif;*.ico;*.jpeg;*.jpg;*.pcx;*.pcc;*.png;*.tga;*.webp", options={'HIDDEN'})
    use_filter: bpy.props.BoolProperty(default=True, options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return addon.connected and context.area.type == 'IMAGE_EDITOR'

    def execute(self, context):
        self.__class__._last_relative = self.relative
        img = context.edit_image
        if img.sb_props.is_sheet:
            img = next((i for i in bpy.data.images if i.sb_props.sheet == img), img)
        img.sb_props.source_set(self.filepath, self.relative)
        msg = encode.sprite_open(name=self.filepath, flags=context.edit_image.sb_props.sync_flags)
        addon.server.send(msg)

        return {'FINISHED'}


    def invoke(self, context, event):
        self.invoke_context = context

        # I have a feeling blender already has a solution but can't seem to find it
        if not hasattr(self.__class__, "_last_relative"):
            self.__class__._last_relative = addon.prefs.use_relative_path
        self.relative = self.__class__._last_relative

        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class SB_OT_sprite_make_animated(bpy.types.Operator):
    bl_idname = "pribambase.sprite_make_animated"
    bl_label = "Enable Animation"
    bl_description = "Mark the image as animated, same as checking Animated in aseprite popup. Takes effect immediately if Aseprite is connected, or next time it connects"
    bl_options = {'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.area.type == 'IMAGE_EDITOR' and context.edit_image and \
            'SHEET' not in context.edit_image.sb_props.sync_flags and not context.edit_image.sb_props.is_sheet
    
    def execute(self, context):
        img = context.edit_image
        img.sb_props.sync_flags = {'SHEET', *img.sb_props.sync_flags}
        msg = encode.sprite_open(img.sb_props.sync_name, img.sb_props.sync_flags)
        addon.server.send(msg)
        return {'FINISHED'}


class SB_OT_sprite_reload_all(bpy.types.Operator):
    bl_idname = "pribambase.sprite_reload_all"
    bl_label = "Reload All Sprites"
    bl_description = "Update data for all sprite textures from their original files"
    bl_options = {'UNDO'}

    @classmethod
    def poll(cls, context):
        return addon.connected

    def execute(self, context):
        images = []

        for img in bpy.data.images:
            if img.sb_props.source:
                if path.exists(img.sb_props.source_abs):
                    images.append((img.sb_props.sync_name, img.sb_props.sync_flags))
                else:
                    self.report({'INFO'}, f"Image {img.name} skipped: file '{img.sb_props.source_abs}' does not exist")

        addon.server.send(encode.peek(images))
        return {'FINISHED'}


class SB_MT_sprite(bpy.types.Menu):
    bl_label = "Sprite"
    bl_idname = "SB_MT_sprite"

    def draw(self, context):
        layout = self.layout

        if not addon.connected:
            layout.operator("pribambase.server_start", icon="ERROR")
            layout.separator()

        layout.operator("pribambase.sprite_new", icon='FILE_NEW')
        layout.operator("pribambase.sprite_open", icon='FILE_FOLDER')
        layout.operator("pribambase.sprite_edit", icon='GREASEPENCIL')
        layout.operator("pribambase.sprite_edit_copy")
        layout.operator("pribambase.sprite_replace")
        layout.separator()
        layout.operator("pribambase.sprite_make_animated")
        layout.operator("pribambase.uv_send", icon='UV_VERTEXSEL')


    def header_draw(self, context):
        # deceiptively, self is not the menu here but the header
        self.layout.menu("SB_MT_sprite")


class SB_PT_sprite(bpy.types.Panel):
    bl_idname = "SB_PT_sprite"
    bl_label = "Sprite"
    bl_category = "Image"
    bl_space_type = "IMAGE_EDITOR"
    bl_region_type = "UI"

    @classmethod
    def poll(cls, context):
        return context.edit_image is not None

    def draw(self, context):
        layout = self.layout
        img = context.edit_image
        props = img.sb_props
        origin = next((i for i in bpy.data.images if i.sb_props.sheet == img), None)

        sprite = layout.row(align=True)
        source = sprite.row()
        if props.is_sheet and origin:
            source.prop(origin.sb_props, "source")
            
            sub = layout.row()
            sub.alignment = 'RIGHT'
            sub.label(text=f"{pgettext('Sheet')} {props.sheet_size[0]}x{props.sheet_size[1]}, {props.animation_length} {pgettext('frames')}")

        else:
            source.prop(context.edit_image.sb_props, "source")
            row = layout.row()
            row.enabled = False
            row.prop(context.edit_image.sb_props, "sheet")

        sprite.operator("pribambase.sprite_purge", icon='TRASH', text="")
        
        layout.row().prop(img.sb_props, "prescale")