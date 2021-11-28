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
import asyncio
import re
import math
from typing import List, Tuple, Iterable
from . import Handler
import numpy as np
# TODO move into local methods
from .. import util


class Batch(Handler):
    """Process batch messages"""
    id = "["

    def parse(self, args):
        count = self.take_uint(2)
        args.messages = [self.take_data() for _ in range(count)]


    async def execute(self, messages:Iterable[memoryview]):
        for m in messages:
            await self._handlers.process(m)


class Image(Handler):
    id = "I"

    def parse(self, args):
        args.size = self.take_uint(2), self.take_uint(2)
        args.frame = self.take_uint(2)
        args.name = self.take_str()
        args.data = np.frombuffer(self.take_data(), dtype=np.ubyte)

    async def execute(self, *, size:Tuple[int, int], frame:int, name:str, data:np.array):
        try:
            # TODO separate cases for named and anonymous sprites
            if not bpy.context.window_manager.is_interface_locked:
                util.update_image(size[0], size[1], name, frame, data)
            else:
                bpy.ops.pribambase.report(message_type='WARNING', message="UI is locked, image update skipped")
        except:
            # blender 2.80... if it crashes, it crashes :\
            util.update_image(size[0], size[1], name, frame, data)


class Spritesheet(Handler):
    """Change textures' sources when aseprite saves the file under a new name"""
    id = 'G'

    def take_tag(self):
        name = self.take_str()
        start = self.take_uint(2)
        end = self.take_uint(2)
        ani_dir = self.take_uint(1)
        return (name, start, end, ani_dir)


    def parse(self, args):
        args.size = self.take_uint(2), self.take_uint(2)
        args.name = self.take_str()
        args.start = self.take_sint(4)
        args.length = self.take_uint(4)
        args.current_frame = self.take_uint(4)
        args.frames = [self.take_frame() for _ in range(args.length)]
        _ntags = self.take_uint(4)
        args.current_tag = self.take_str()
        args.tags = [self.take_tag() for _ in range(_ntags)]
        args.images = [self.take_data() for _ in range(args.length)]


    async def execute(self, *, size:Tuple[int, int], name:str, start:int, length:int, frames:List[int], tags:List[Tuple], current_frame:int, current_tag:str, images:List[np.array]):
        count_x = math.ceil(length ** 0.5)
        count_y = math.ceil(length / count_x)
        w, h = size
        stride = w * 4

        # TODO profile if changing to .empty gives significant perf (at cost of messy look)
        sheet_data = np.zeros((h * count_y, w * count_x * 4), dtype=np.ubyte)

        for n,frame in enumerate(images):
            # TODO is there a way to just swap the nparray's buffer? 
            x, y = n % count_x, n // count_x
            fd = np.frombuffer(frame, dtype=np.ubyte)
            fd.shape = (h, stride)
            dst = sheet_data[y * h : (y + 1) * h, x * stride: (x + 1) * stride]
            np.copyto(dst, fd, casting='no')

        try:
            if not bpy.context.window_manager.is_interface_locked:
                util.update_spritesheet(size, (count_x, count_y), name, start, frames, tags, current_frame, current_tag, sheet_data)
            else:
                bpy.ops.pribambase.report(message_type='WARNING', message="UI is locked, image update skipped")
        except:
            # version 2.80... caveat emptor
            util.update_spritesheet(size, (count_x, count_y), name, start, frames, tags, current_frame, current_tag, sheet_data)


class Frame(Handler):
    """Change sprite frame without changing data"""
    id = "F"

    def parse(self, args):
        args.frame = self.take_uint(4)
        args.name = self.take_str()
        args.start = self.take_uint(2)
        nframes = self.take_uint(4)
        args.frames = [self.take_frame() for _ in range(nframes)]


    async def execute(self, frame:int, name:str, start:int, frames:List[Tuple[int, int]]):
        try:
            if not bpy.context.window_manager.is_interface_locked:
                util.update_frame(name, frame, start, frames)
            else:
                bpy.ops.pribambase.report(message_type='WARNING', message="UI is locked, frame flip skipped")
        except:
            # version 2.80... caveat emptor
            util.update_frame(name, frame, start, frames)


class ChangeName(Handler):
    """Change textures' sources when aseprite saves the file under a new name"""
    id = "C"

    def parse(self, args):
        args.old_name = self.take_str()
        args.new_name = self.take_str()


    async def execute(self, *, old_name, new_name):
        try:
            # FIXME there's a risk of race condition but it's pretty bad if the rename doesn't happen
            while bpy.context.window_manager.is_interface_locked:
                bpy.ops.pribambase.report(message_type='WARNING', message="UI is locked, waiting to update image source..")
                asyncio.sleep(0.1)
        except:
            # version 2.80... caveat emptor
            pass

        # avoid having identical sb_source on several images
        for img in bpy.data.images:
            if old_name in (img.sb_props.source_abs, img.filepath, img.name):
                img.sb_props.source_set(new_name)

                if re.search(r"\.(?:png|jpg|jpeg|bmp|tga)$", new_name):
                    img.filepath = new_name
                else:
                    img.filepath = ""

                bpy.ops.pribambase.texture_list()
