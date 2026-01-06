from PIL import Image
import numpy as np
import math

##USE Left handed coordinates

class Object3D:
    def __init__(self, filename = None, pos = np.array([0.0, 0.0, 0.0, 1]), rot = np.array([0.0, 0.0, 0.0])):
        if filename is None:
            # 8 vertices of the unit cube [x, y, z]
            self.v = np.array([
                [0, 0,  0, 1],   # 0
                [0, 0, -1, 1],   # 1  ← Z flipped!
                [0, 1,  0, 1],   # 2
                [0, 1, -1, 1],   # 3
                [1, 0,  0, 1],   # 4
                [1, 0, -1, 1],   # 5
                [1, 1,  0, 1],   # 6
                [1, 1, -1, 1]    # 7
            ], dtype=float)
            
            # UV coordinates stay the same
            self.vt = np.array([
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1]
            ], dtype=float)
            
            # Triangles with REVERSED winding (now clockwise for left-handed)
            self.f = np.array([
                # -x face
                [0, 6, 2], [0, 4, 6],
                # +x face
                [1, 7, 5], [1, 3, 7],
                # -y face
                [0, 5, 4], [0, 1, 5],
                # +y face
                [2, 3, 7], [2, 7, 6],
                # -z face (now the near face)
                [0, 1, 3], [0, 3, 2],
                # +z face (now the far face)
                [4, 7, 6], [4, 5, 7]
            ], dtype=int)
            self.transform_matrix = np.eye(4, dtype=float)
            self.rotation_matrix = np.eye(4, dtype=float)
            if pos is not None:
                self.pos = pos
            else:
                self.pos = np.array([0.0, 0.0, 0.0, 1])
            if rot is not None:
                self.rot = rot
            else:
                self.rot = np.array([0.0, 0.0, 0.0])#roll pitch yaw
        else:
            pass
        self._update_transform()

    def _update_transform(self):
        """Rebuild the 4x4 transformation matrix from position and rotation"""
        # Start fresh
        transform = np.eye(4)
        
        # Translation
        #print(np.array([self.pos]).transpose())
        self.transform_matrix[:, 3] = np.array(self.pos).transpose()
        # Rotation (X -> Y -> Z order)
        cx, sx = np.cos(self.rot[0]), np.sin(self.rot[0])
        cy, sy = np.cos(self.rot[1]), np.sin(self.rot[1])
        cz, sz = np.cos(self.rot[2]), np.sin(self.rot[2])
        
        # Rotation matrix
        Rx = np.array([[1, 0, 0],
                       [0, cx, -sx],
                       [0, sx, cx]])
        Ry = np.array([[cy, 0, sy],
                       [0, 1, 0],
                       [-sy, 0, cy]])
        Rz = np.array([[cz, sz, 0],
                       [-sz, cz, 0],
                       [0, 0, 1]])
        
        rotation_matrix = Rz @ Ry @ Rx
        self.rotation_matrix[:3, :3] = rotation_matrix

    def translate(self, dx=0, dy=0, dz=0):
        """Move the object by given amounts"""
        self.pos += np.array([dx, dy, dz, 0])
        self._update_transform()

    def set_position(self, x=0, y=0, z=0):
        """Set absolute position"""
        self.pos = np.array([x, y, z, 1])
        self._update_transform()

    def rotate(self, rx=0.0, ry=0.0, rz=0.0, degrees=False):
        """Add rotation (in radians by default)"""
        if degrees:
            rx, ry, rz = np.radians([rx, ry, rz])
        self.rot += np.array([rx, ry, rz])
        self._update_transform()

    def set_rotation(self, rx=0.0, ry=0.0, rz=0.0, degrees=False):
        """Set absolute rotation"""
        if degrees:
            rx, ry, rz = np.radians([rx, ry, rz])
        self.rot = np.array([rx, ry, rz])
        self._update_transform()
    
    def get_model_to_world_matrix(self):
        return np.matmul(self.transform_matrix, self.rotation_matrix)

class Camera(Object3D):
    def __init__(self, resolution=(320,240), fov=(90,74), pos = [0,0,-5,1], facing=[0,0,1], rot=[0,0,0]): #neutral is pointing +z
        #rot direction. rotation axis pointing inwards to viewer, the rotation direction is ccw
        super().__init__(pos=pos, rot=np.array(rot))
        self.resolution = resolution
        self.fov=fov
        self.facing = facing
        self.translate_inverse = np.eye(4)
        self.rotate_inverse = np.eye(4)
        self._update_transform()

    def _update_transform(self):
        super()._update_transform()
        self.translate_inverse = np.array([
            [1, 0, 0, -self.pos[0]],
            [0, 1, 0, -self.pos[1]],
            [0, 0, 1, -self.pos[2]],
            [1, 0, 0, 1]])
        self.rotate_inverse = self.rotation_matrix.transpose()
    
    def _project(self, vertexes): #vertexes must needs to be in View coordinates
        d=1
        proj = np.zeros_like(vertexes[:, :2])
        for i in range(len(proj)):
            proj[i][0] = vertexes[i][0]/vertexes[i][2]
            proj[i][1] = vertexes[i][1]/vertexes[i][2]

        return proj
    
    def _filter_visable_face_wireframe(self, vertex_view, vertex_project, face):
        visable_x = 1
        visable_y = 0.754
        visable_lines = []
        for i in range(len(list(face))):
            for j in range(len(list(face[i]))):
                if vertex_view[face[i][j]][2]<=0 or vertex_view[face[i][(j+1)%3]][2]<=0:
                    continue
                if abs(vertex_project[face[i][j]][0]) >= visable_x or abs(vertex_project[face[i][(j+1)%3]][0]) >= visable_x:
                    continue
                elif abs(vertex_project[face[i][j]][1]) >= visable_y or abs(vertex_project[face[i][(j+1)%3]][1]) >= visable_y:
                    continue
                else:
                    visable_lines.append((int(face[i][j]), int(face[i][(j+1)%3])))
        #print(len(visable_lines))
        return visable_lines

    def _normalize_projected_vertex(self,v_proj):
        norm_vertex = []
        scale_factor_x = self.resolution[0]/math.tan(self.fov[0])
        scale_factor_y = -scale_factor_x
        trans_x = self.resolution[0]/2
        trans_y = self.resolution[1]/2
        for v in v_proj:
            norm_vertex.append((int(v[0]*scale_factor_x + trans_x), int(v[1]*scale_factor_y + trans_y)))
        return norm_vertex
    
    def _draw_wireframe(self, norm_vertex, line_visable):
        norm = self.resolution[0]
        img = Image.new('RGB', (self.resolution[0],self.resolution[1]), color='white')
        pixels = img.load()
        for x in range(320):
            for y in range(240):
                pixels[x, y] = (0, 0, 0) 
        #print(line_visable)
        #print(len(line_visable))
        for f in range(len(line_visable)):
            line = line_visable[f]
            #print(line)
            dx = norm_vertex[line[1]][0] - norm_vertex[line[0]][0]
            dy = norm_vertex[line[1]][1] - norm_vertex[line[0]][1]
            if dx==0 and dy==0:
                Dx=0
                Dy=0
            elif dx==0:
                Dy=0
                Dx=100
            elif dy==0:
                Dx=0
                Dy=100
            else:
                Dx = dy/dx
                Dy = dx/dy
            #print('dxdy')
            #print([Dx,Dy, dx, dy])
            if abs(Dx)<=1:
                for i in range(0,int(dx),int(math.copysign(1,dx))):
                    #print(int(vertex_proj[line[0]][0]*norm + i))
                    #try:
                    pixels[int(norm_vertex[line[0]][0] + i), int(norm_vertex[line[0]][1] + i*Dx)] = (150, 150, 150)
                    #except IndexError:
                    #    pass
                        #print("Invalid index", int(vertex_proj[line[0]][0]*norm + i)+160+1, -int(vertex_proj[line[0]][1]*norm + i*Dx)+120-1)
            else:
                for i in range(0,int(dy),int(math.copysign(1,dy))):
                    #try:
                    pixels[int(norm_vertex[line[0]][0] + i*Dy), int(norm_vertex[line[0]][1] + i)] = (150, 150, 150)
                    #except IndexError:
                    #    pass
                        #print("Invalid index",int(vertex_proj[line[0]][0]*norm + i*Dy)+160+1, -int(vertex_proj[line[0]][1]*norm + i)+120-1)
        
        print(f)

        return img
    def _draw_wireframe2(self, vertex_proj, line_visable):
        norm = self.resolution[0]
        img = Image.new('RGB', (self.resolution[0],self.resolution[1]), color='white')
        pixels = img.load()
        for x in range(320):
            for y in range(240):
                pixels[x, y] = (0, 0, 0) 
        #print(line_visable)
        #print(len(line_visable))
        for f in range(len(line_visable)):
            line = line_visable[f]
            #print(line)
            dx = vertex_proj[line[1]][0] - vertex_proj[line[0]][0]
            dy = vertex_proj[line[1]][1] - vertex_proj[line[0]][1]
            Dx = dy/dx
            Dy = dx/dy
            if dx==0 and dy==0:
                Dx=0
                Dy=0
            #print('dxdy')
            #print([Dx,Dy, dx, dy])
            if abs(Dx)<=1:
                for i in range(int(dx*norm)+1):
                    #print(int(vertex_proj[line[0]][0]*norm + i))
                    try:
                        pixels[int(vertex_proj[line[0]][0]*norm + i)+160+1, -int(vertex_proj[line[0]][1]*norm + i*Dx)+120-1] = (150, 150, 150)
                    except IndexError:
                        pass
                        #print("Invalid index", int(vertex_proj[line[0]][0]*norm + i)+160+1, -int(vertex_proj[line[0]][1]*norm + i*Dx)+120-1)
            else:
                for i in range(int(dy*norm)+1):
                    try:
                        pixels[int(vertex_proj[line[0]][0]*norm + i*Dy)+160+1, -int(vertex_proj[line[0]][1]*norm + i)+120-1] = (150, 150, 150)
                    except IndexError:
                        pass
                        #print("Invalid index",int(vertex_proj[line[0]][0]*norm + i*Dy)+160+1, -int(vertex_proj[line[0]][1]*norm + i)+120-1)
        
        print(f)

        return img


    def get_world_to_view_matrix(self):
        return np.matmul(self.rotate_inverse, self.translate_inverse)
    
    def snapshot(self, obj, out = False):
        if 1==1:
            model_to_world = obj.get_model_to_world_matrix()
            #print(model_to_world)
            world_to_view = self.get_world_to_view_matrix()
            #print(world_to_view)
            obj_vertex_world = np.matmul(model_to_world, obj.v.transpose())
            #print(obj_vertex_world)
            obj_vertex_view = np.matmul(world_to_view, obj_vertex_world)
            #print(obj_vertex_view)
            projected_vertex = self._project(obj_vertex_view.transpose())
            #print('v_proj')
            #print(projected_vertex)
            #print(obj_vertex_view, projected_vertex, obj_vertex_world)
            visable_lines = self._filter_visable_face_wireframe(obj_vertex_view.transpose(), projected_vertex, obj.f)
            #print('visable_lines')
            #print(visable_lines)
            #print({i for i in visable_lines})
            proj_norm_vertex = self._normalize_projected_vertex(projected_vertex)
            #print(proj_norm_vertex)
            img = self._draw_wireframe(proj_norm_vertex, visable_lines)

            if out:
                img.show()

            return img

        elif obj is list:
            pass
        

obj = Object3D()
cam = Camera()

#obj.translate(dx=-0.5, dy= -0.5)
#obj.rotate(rz=90, degrees=True)

gif = []

#'''
for i in range(360):
    obj.rotate(ry=2, rz=1, degrees=True)
    img = cam.snapshot(obj)
    gif.append(img)
gif[0].save(
    'rotationyz.gif',
    save_all=True,
    append_images=gif[1:],
    duration=1,    # adjust this for speed~ faster = smaller number!
    loop=0
)
#'''

#cam.snapshot(obj, out = True)
