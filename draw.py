from os import name
import bpy
import gpu
from bpy_extras import view3d_utils
from mathutils import Vector
from mathutils.geometry import intersect_point_tri, barycentric_transform
from gpu_extras.batch import batch_for_shader

from typing import Collection, Generator, Tuple

"""
Simple Ink
Alpha Compositing
Copy Alpha + Color
Lock Alpha
"""

blend_modes = (
    ('SIMPLE', "Simple Ink", ( 
        "If the foreground color is opaque (alpha = 1), it paints with the given opaque color. "
        "If the color has alpha (0 < alpha < 1), it composite the color with the layer surface. "
        "If the color is transparent (alpha = 0), the tool acts like an Eraser")),
    ('ALPHA', "Alpha Compositing", "It merges the foreground color with the layer surface depending on the alpha value of the foreground color"),
    ('COPY', "Copy Alpha+Color", "It replaces the layer surface pixels with the active foreground color with its alpha value. It doesn't make any kind of alpha compositing, it just takes the active color and put it exactly as it is in the destination pixel"),
    ('LOCK', "Lock Alpha", "In this case the original alpha values from the layer surface are kept, and only the RGB color components are replaced from the foreground color"))

brush_shapes = (
    ('ROUND', "Round", ""),
    ('SQUARE', "Square", ""))


def line(x1,y1,x2,y2):
    """Bresenham line stepping generator"""
    w = x2 - x1
    w_abs = abs(w)
    h = y2 - y1
    h_abs = abs(h)
    x = x1
    y = y1
    dx = -1 if w < 0 else 1
    dy = -1 if h < 0 else 1

    if abs(w) > abs(h):
        yield x,y

        pk = 2 * h_abs - w_abs
        for i in range(0, w_abs):
            x += dx
            if pk < 0:
                pk += 2 * h_abs
            else:
                y += dy
                pk += 2 * h_abs - 2 * w_abs
            
            yield x, y
    
    else:
        yield x,y

        pk = 2 * w_abs - h_abs

        for i in range(0, h_abs):
            y += dy
            if pk < 0:
                pk += 2 * w_abs
            else:
                x += dx
                pk += 2 * w_abs - 2 * h_abs
            
            yield x, y

def draw_replace(img:bpy.types.Image, dots:Generator[Tuple[int, int], None, None], color:Collection[float], *, replace_alpha=True):
    pix = img.pixels
    for x, y in dots:
        addr = 4 * (img.size[0] * y + x)
        pix[addr + 0] = color[0]
        pix[addr + 1] = color[1]
        pix[addr + 2] = color[2]
        if replace_alpha:
            pix[addr + 3] = color[3]


def draw_alpha(img:bpy.types.Image, dots:Generator[Tuple[int, int], None, None], color:Collection[float]):
    if color[3] == 0: 
        return

    pix = img.pixels
    for x, y in dots:
        addr = 4 * (img.size[0] * y + x)
        a = color[3]
        da = a * (1 - pix[addr + 3]) # no particular meaning, just repeated a lot
        outa = pix[addr + 3] + da

        pix[addr] = (pix[addr] * a + color[0] * da) / outa
        pix[addr + 1] = (pix[addr + 1] * a + color[1] * da) / outa
        pix[addr + 2] = (pix[addr + 2] * a + color[2] * da) / outa
        pix[addr + 3] = outa


def draw_simple(img:bpy.types.Image, dots:Generator[Tuple[int, int], None, None], color:Collection[float]):
    if color[3] == 1.0:
        draw_replace(img, dots, color, replace_alpha=True)
    elif color[3] == 0.0:
        # zero alpha non-zero rgb tends to cause weird issues when rendering/gamengines/etc
        draw_replace(img, dots, (0, 0, 0, 0), replace_alpha=True)
    else:
        draw_alpha(img, dots, color)


def _draw_callback_px(self, context):
    if not self.brush:
        return

    shader = gpu.shader.from_builtin('3D_UNIFORM_COLOR')
    batch = batch_for_shader(shader, 'TRI_STRIP', {"pos": self.brush})
    shader.bind()
    shader.uniform_float("color", (0.0, 1.0, 0.72, 1))
    batch.draw(shader)


class SB_OT_pencil(bpy.types.Operator):
    bl_description = "..."
    bl_idname = "pribambase.draw_with_pencil"
    bl_label = "Pencil"
    bl_options = {'REGISTER', 'UNDO'}
    bl_context_mode = 'PAINT_TEXTURE'


    fg: bpy.props.FloatVectorProperty(
        name="Color", 
        description="Foreground Color (left click)", 
        size=4, 
        subtype='COLOR',
        default=(0.8, 0.8, 0.8, 1.0),
        min=0.0,
        max=1.0)

    fg_mode: bpy.props.EnumProperty(
        name="Color Mode", 
        description="Foreground color blending type", 
        items=blend_modes)

    bg: bpy.props.FloatVectorProperty(
        name="Background", 
        description="Background Color (right click)", 
        size=4, 
        subtype='COLOR',
        default=(0.2, 0.2, 0.2, 1.0),
        min=0.0,
        max=1.0)

    bg_mode: bpy.props.EnumProperty(
        name="Background Mode", 
        description="Background color blending type", 
        items=blend_modes)

    size: bpy.props.IntProperty(name="Size", 
        description="Brush diameter",
        default=1,
        min=1,
        max=8) # bigger than that is slow - use the default brush

    shape: bpy.props.EnumProperty(
        name="Shape", 
        description="Brush shape", 
        items=brush_shapes)

    ### 
    draw_with_bg: bpy.props.BoolProperty(default=False, options={'HIDDEN'})

    @classmethod
    def poll(cls, context:bpy.types.Context):
        obj:bpy.types.Object = context.image_paint_object
        return obj and obj.active_material.texture_paint_images[obj.active_material.paint_active_slot]
    

    def cast_ray(self, context, event):
        """Raycast mouse position and find uv of the hit; or None"""
        region = context.region
        rv3d = context.region_data
        coord = event.mouse_region_x, event.mouse_region_y
        obj = self.obj

        # get the ray from the viewport and mouse
        view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
        ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        ray_target = ray_origin + view_vector

        matrix_inv = obj.matrix_world.inverted()
        ray_origin_obj = matrix_inv @ ray_origin
        ray_target_obj = matrix_inv @ ray_target
        ray_direction_obj = ray_target_obj - ray_origin_obj

        success, location, normal, face_index = obj.ray_cast(ray_origin_obj, ray_direction_obj)

        if success:
            mesh = self.obj.data
            for tri in mesh.loop_triangles:
                # need to use barycentric transform (triangle <-> triangle, no polys)
                # hence check which triangle we've hit, cuz texture stretch can be different
                if tri.polygon_index == face_index:
                    verts = [mesh.vertices[v].co for v in tri.vertices]
                    if intersect_point_tri(location, *verts):
                        uvs = [(*mesh.uv_layers.active.data[i].uv, 0) for i in tri.loops]
                        location_uv = Vector(barycentric_transform(location, *verts, *uvs))
                        return location_uv, uvs, verts
        return None, None, None


    def modal(self, context, event:bpy.types.Event):
        if event.type == 'MOUSEMOVE':
            location_uv, uvs, verts = self.cast_ray(context, event)
            if location_uv is None:
                self.brush = []
            else:
                # pixel corners; 2d vectors, with Z=0 added to transform to 3d later
                px = (location_uv[0] - location_uv[0] % self.grid[0], location_uv[1] - location_uv[1] % self.grid[1], 0)

                # set brush preview
                strip = [px,
                    (px[0], px[1] + self.grid[1], 0),
                    (px[0] + self.grid[0], px[1] + self.grid[1], 0),
                    (px[0] + self.grid[0], px[1], 0), 
                    px]
                self.brush = [barycentric_transform(p, *uvs, *verts) for p in strip]

                # draw
                if self.is_drawing:
                    x = int(px[0] / self.grid[0])
                    y = int(px[1] / self.grid[1])
                    
                    x0, y0 = self.last_px
                    if x0 != x or y0 != y:
                        color = self.bg if self.draw_with_bg else self.fg
                        mode = self.bg_mode if self.draw_with_bg else self.fg_mode

                        if mode == 'SIMPLE':
                            draw_simple(self.image, line(x0, y0, x, y), color)
                        elif mode == 'ALPHA':
                            draw_alpha(self.image, line(x0, y0, x, y), color)
                        elif mode == 'COPY':
                            draw_replace(self.image, line(x0, y0, x, y), color, replace_alpha=True)
                        elif mode == 'LOCK':
                            draw_replace(self.image, line(x0, y0, x, y), color, replace_alpha=False)
                        self.image.update()
                        self.image.update_tag()
                        self.last_px = x, y
            context.area.tag_redraw()
        
        elif event.type in ('LEFTMOUSE', 'RIGHTMOUSE'):
            if event.value == 'PRESS':
                self.is_drawing = True
                location_uv, _, __ = self.cast_ray(context, event)
                if location_uv is not None:
                    self.last_px = (int(location_uv.x / self.grid[0]), int(location_uv.y / self.grid[1]))

            elif event.value == 'RELEASE':
                # over
                self.is_drawing = False
                self.image.update_tag()
                context.window.cursor_modal_restore()
                bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
                context.area.tag_redraw()
                return {'FINISHED'}

        return {'RUNNING_MODAL'}


    def invoke(self, context:bpy.types.Context, event):
        self.obj:bpy.types.Object = bpy.context.image_paint_object
        self.obj.data.calc_loop_triangles()
        self.image = self.obj.active_material.texture_paint_images[self.obj.active_material.paint_active_slot]
        self.grid = (1/self.image.size[0], 1/self.image.size[1], 0)
        self.brush = None
        self.last_px = (-1, -1)
        self.is_drawing = False

        # add handlers
        args = (self, context)
        self._handle = bpy.types.SpaceView3D.draw_handler_add(_draw_callback_px, args, 'WINDOW', 'POST_VIEW')
        context.window_manager.modal_handler_add(self)

        # when executing as tool, the first mouse click is not caught by modal
        # (could've copied pen down code here instead)
        self.modal(context, event)

        # change cursor
        context.window.cursor_modal_set('CROSSHAIR')

        return {'RUNNING_MODAL'}


class SB_WT_Pencil(bpy.types.WorkSpaceTool):
    bl_space_type = 'VIEW_3D'
    bl_context_mode = 'PAINT_TEXTURE'

    # The prefix of the idname should be your add-on name.
    bl_idname = "pribambase.pencil_tool"
    bl_label = "Pencil"
    bl_description = "Draw image pixels"
    bl_keymap = (
        ("pribambase.draw_with_pencil", {"type": 'LEFTMOUSE', "value": 'PRESS'}, None),
        ("pribambase.draw_with_pencil", {"type": 'RIGHTMOUSE', "value": 'PRESS'}, {"properties": [("draw_with_bg", True)]}))

    def draw_settings(context, layout, tool):
        props = tool.operator_properties("pribambase.draw_with_pencil")

        layout.prop(props, "size")
        layout.prop(props, "shape")

        row = layout.row(align=True)
        row.prop(props, "fg")
        row.prop(props, "fg_mode", text="")

        row = layout.row(align=True)
        row.prop(props, "bg", text=" ")
        row.prop(props, "bg_mode", text="")