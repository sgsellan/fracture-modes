# Include existing libraries
import numpy as np
import igl
import tetgen
# Include my own general functionality
from context import fracture_utility as fracture
from context import gpytoolbox
from gpytoolbox.copyleft import lazy_cage

# This is the "fine mesh", i.e. the mesh we use for rendering
v_fine, f_fine = igl.read_triangle_mesh("data/bunny_oded.obj")
v_fine = gpytoolbox.normalize_points(v_fine)
# This is the "cage mesh", i.e. the coarser mesh that we will tetrahedralize and use for the physical simulation
v, f = lazy_cage(v_fine,f_fine,num_faces=2000)

tgen = tetgen.TetGen(v,f)
nodes, elements =  tgen.tetrahedralize()

# Initialize fracture mode class
modes = fracture.fracture_modes(nodes,elements) 
# Set parameters for call to fracture modes
params = fracture.fracture_modes_parameters(num_modes=10,verbose=True,d=3)
# Compute fracture modes
modes.compute_modes(parameters=params)
modes.impact_precomputation(v_fine=v_fine,f_fine=f_fine)
modes.write_segmented_modes("output_modes")

contact_point = nodes[1,:]
direction = np.array([1.0,0.0,0.0])
# First projection, this should be fast
modes.impact_projection(contact_point=contact_point,direction=direction)
# Second projection, this should be fast
new_contact_point = nodes[5,:]
modes.impact_projection(contact_point=new_contact_point,direction=direction)
# Write segmented output to obj
modes.write_segmented_output("output.obj")
print("Example ran successfully!")