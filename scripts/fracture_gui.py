# Include existing libraries
import os
import numpy as np
import igl
import polyscope as ps
import polyscope.imgui as psim
from scipy.sparse import kron, eye
import tetgen
from datetime import datetime
# Include my own functionality
from context import fracture_utility as fracture
from context import gpytoolbox
from context import sys

from gpytoolbox.copyleft import lazy_cage


filename = "data/bunny_oded.obj"
if len(sys.argv)>1:
    filename = sys.argv[1]

print("Loading GUI...")

v, f = igl.read_triangle_mesh(filename)
v = gpytoolbox.normalize_points(v)

# Generate some random points as impacts
B,FI = igl.random_points_on_mesh(1000,v,f)
B = np.vstack((B[:,0],B[:,0],B[:,0],B[:,1],B[:,1],B[:,1],B[:,2],B[:,2],B[:,2])).T
P = B[:,0:3]*v[f[FI,0],:] + B[:,3:6]*v[f[FI,1],:] + B[:,6:9]*v[f[FI,2],:]
n = igl.per_face_normals(v,f,np.array([[0.0,0.0,0.0]]))
N = n[FI,:]

params = fracture.fracture_modes_parameters(num_modes=3,verbose=True,d=1)



##### GUI
show_full_impact = False
impact_text = ""
modes_text = ""
computed_modes = False
showing_modes = False
showing_input = True
showing_impact = False
ps_vol = []
t = 0.0
face_num = 2000
gs = 50
v_fine = v
f_fine = f
v, f = lazy_cage(v_fine,f_fine,num_faces=face_num,grid_size = gs)
ind = 0
threshold = 10
now = datetime.now()
dt_string = now.strftime("_%d_%m_%H_%M")
base=os.path.basename(filename)
write_obj_name =  os.path.splitext(base)[0] + dt_string + ".obj"
write_modes_name = os.path.splitext(base)[0]

ps.init()
ps.set_ground_plane_mode("none")
ps.set_transparency_mode('pretty')
ps.set_transparency_render_passes(2)
ps.set_program_name("Fracture Modes GUI")
ps_input_mesh = ps.register_surface_mesh("input mesh", v_fine, f_fine)
ps_cage = ps.register_surface_mesh("cage mesh", v, f)
ps_cage.set_transparency(0.4)

def callback():
    global t, ind, nodes, elements, params, computed_modes, showing_modes, showing_input, v, f, ps_input_mesh, ps_vol, modes, off, UU, face_num, ps_cage, v_fine, f_fine, ps_impact_mesh, ps_fracture_mesh,P,ind, showing_impact, ps_impact_projected_mesh, threshold, impact, impact_text, modes_text, off_x, off_y, contact_point, gs, modes_1d, labels_fine_1d, labels_1d, fine_vertices_1d, fine_triangles_1d, off_tets_x, off_tets, off_tets_x_exploded, off_tets_y, direction, show_full_impact
    # Executed every frame
    # Do computation here, define custom UIs, etc.
    changed, params.num_modes = psim.InputInt("Number of modes", params.num_modes, step=1, step_fast=10) 

    if(psim.Button("Compute modes")):
        tgen = tetgen.TetGen(v,f)
        nodes, elements =  tgen.tetrahedralize(minratio=1.4)
        modes_text = "Computing modes..."
        modes = fracture.fracture_modes(nodes,elements)
        modes.compute_modes(params)
        modes.impact_precomputation(v_fine=v_fine,f_fine=f_fine)
        labels_fine_1d = modes.fine_labels.copy()
        fine_vertices_1d = modes.fine_vertices.copy()
        fine_triangles_1d = modes.fine_triangles.copy()
        modes_1d = modes.modes.copy() # For visualization only
        labels_1d = modes.labels.copy() # For visualization only
        # modes.transfer_modes_to_3d()
        # modes.impact_precomputation(v_fine=v_fine,f_fine=f_fine)
        UU = modes_1d.copy() # For visualization only
        UU = np.vstack((np.zeros((2*UU.shape[0],UU.shape[1])),UU)) # Make Z be the dim
        off = 0.*modes.fine_vertices
        off_x = off.copy()
        off_y = off.copy()
        off_x[:,0] = 1.2
        off_y[:,2] = 1.2
        computed_modes = True
        modes_text = "Computed modes"
        t = 0.0


    if computed_modes:
        if (psim.Button("Write segmented modes")):
            modes.write_segmented_modes(write_modes_name)

    psim.Text(modes_text)
    if(psim.Button("Show modes")):
        if showing_input:
            ps_input_mesh.remove()
            ps_cage.remove()
            showing_input = False
        ps_vol = []
        ps.set_transparency_mode('none')
        off = 0.*fine_vertices_1d
        off_x = off.copy()
        off_y = off.copy()
        off_x[:,0] = 1.2
        off_y[:,2] = 1.2
        for i in range(UU.shape[1]):
            mesh_name = "Mode number {}".format(i)
            scalar_map_name = "Labels for mode number {}".format(i)
            ps_vol.append(ps.register_surface_mesh(mesh_name,vertices=fine_vertices_1d + (i//4)*off_y +  (i%4)*off_x, faces=fine_triangles_1d))
            ps_vol[i].add_scalar_quantity(scalar_map_name, labels_fine_1d[:,i],enabled=True,cmap='phase',vminmax=(0,np.max(labels_fine_1d[:,i])+1.0))
        ps.reset_camera_to_home_view()
        showing_modes = True
    if(psim.Button("Show input")):
        if showing_modes:
            for i in range(len(ps_vol)):
                ps_vol[i].remove()
            showing_modes = False
        if showing_impact:
            ps_impact_mesh.remove()
            ps_fracture_mesh.remove()
        if not showing_input:
            ps_input_mesh = ps.register_surface_mesh("input mesh", v_fine, f_fine)
            ps_cage = ps.register_surface_mesh("cage mesh", v, f)
            ps_cage.set_transparency(0.4)
            showing_input = True

    # if showing_modes:
    #     t = t+0.1
    #     for i in range(len(ps_vol)):
    #         ps_vol[i].update_vertex_positions(modes.fine_vertices + i*off_x + 0.1*np.sin(t)*np.reshape(UU[:,i],modes.fine_vertices.shape,order='F'))



    changed, face_num = psim.InputInt("Faces in cage", face_num)
    if changed:
        v, f = lazy_cage(v_fine,f_fine,num_faces=face_num,grid_size = gs)
        ps_cage = ps.register_surface_mesh("cage mesh", v, f)
        ps_cage.set_transparency(0.4)

    ## Impact stuff

    if(psim.Button("Impact mode")):
        ps.set_transparency_mode('none')
        showing_impact = True
        if showing_modes:
            for i in range(len(ps_vol)):
                ps_vol[i].remove()
            showing_modes = False
        if showing_input:
            ps_input_mesh.remove()
            ps_cage.remove()
            showing_input = False
        off = 0.*modes.fine_vertices
        off_x = off.copy()
        off_y = off.copy()
        off_x[:,0] = 1.2
        off_y[:,2] = 1.2
        off_tets = 0.*modes.vertices
        off_tets_x = off_tets.copy()
        off_tets_x[:,0] = 1.2
        off_tets_exploded = 0.*modes.exploded_vertices
        off_tets_x_exploded = off_tets_exploded.copy()
        off_tets_x_exploded[:,0] = 1.2
        ps_impact_mesh = ps.register_volume_mesh("impact mesh", modes.vertices, tets=modes.elements)
        if show_full_impact:
            ps_impact_projected_mesh = ps.register_volume_mesh("projected impact mesh", modes.exploded_vertices - off_tets_x_exploded, tets=modes.exploded_elements)
        else:
            ps_impact_projected_mesh = ps.register_volume_mesh("projected impact mesh", modes.vertices - off_tets_x, tets=modes.elements)
        ps_fracture_mesh = ps.register_surface_mesh("fracture mesh", modes.fine_vertices - 2*off_x, modes.fine_triangles)
        ps.reset_camera_to_home_view()

    if showing_impact:
        changed, threshold = psim.SliderFloat("Threshold", threshold, v_min=0.0, v_max=1000.0)
        if changed and modes.impact_projected:
            modes.impact_projection(contact_point=contact_point,threshold=threshold,direction=direction)
            ps_fracture_mesh.add_scalar_quantity("fracture", modes.fine_vertex_labels_after_impact, enabled=True)
        if(psim.Button("Generate random impact")):
            contact_point = P[ind,:]
            # direction = N[ind,:]
            direction = np.array([1])
            ind = ind + 1
            modes.impact_projection(contact_point=contact_point,threshold=threshold,direction=direction)
            if show_full_impact:
                ps_impact_mesh.update_vertex_positions(modes.vertices + 10*np.reshape(modes.impact_vis,(-1,3)))
                ps_impact_projected_mesh.update_vertex_positions(modes.exploded_vertices - off_tets_x_exploded + 0.002*modes.tet_to_vertex_matrix @ modes.piece_to_tet_matrix @ np.reshape( modes.projected_impact,(-1,3),order='F'))
            else:
                if direction.shape[0]==1:
                    ps_impact_mesh.add_scalar_quantity("impact", np.squeeze(modes.impact_vis), enabled=True, defined_on='vertices')
                    ps_impact_projected_mesh.add_scalar_quantity("projected impact", modes.piece_to_tet_matrix @ modes.projected_impact, enabled=True, defined_on='cells')
                else:
                    ps_impact_mesh.add_scalar_quantity("impact", np.linalg.norm(np.reshape(modes.impact_vis,(-1,3),order='F'),axis=1), enabled=True, defined_on='vertices')
                    ps_impact_projected_mesh.add_scalar_quantity("projected impact", np.linalg.norm(modes.piece_to_tet_matrix @ np.reshape(modes.projected_impact,(-1,3),order='F'),axis=1), enabled=True, defined_on='cells')
            ps_fracture_mesh.add_scalar_quantity("fracture", modes.fine_vertex_labels_after_impact, enabled=True)
            impact_text = "Impact time: {} ms".format(modes.t_impact*1000)
        if(psim.Button("Save segmented output")):
            modes.write_segmented_output(write_obj_name)
    
    psim.Text(impact_text)



ps.set_user_callback(callback)
ps.show()
