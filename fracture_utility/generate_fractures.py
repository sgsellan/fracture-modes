# Include existing libraries
from nbformat import write
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, eye, kron
from scipy.sparse.linalg import lsqr, spsolve
from scipy.sparse.csgraph import connected_components
# Libigl
import igl
import tetgen
from scipy.stats import multivariate_normal


import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../ext/gpytoolbox')))
import gpytoolbox

from .fracture_modes_parameters import fracture_modes_parameters
from .fracture_modes import fracture_modes


def generate_fractures(input_dir,num_modes=20,num_impacts=80,output_dir=None,verbose=True,compressed=True):

    # directory = os.fsencode(input_dir)
    
    # for file in os.listdir(directory):
    filename = input_dir
    t0 = time.time()
    try:
        t00 = time.time()
        v_fine, f_fine = igl.read_triangle_mesh(filename)
        # Let's normalize it so that parameter choice makes sense
        v_fine = gpytoolbox.normalize_points(v_fine)
        t01 = time.time()
        reading_time = t01-t00
        if verbose:
            print("Read shape in",round(reading_time,3),"seconds.")
        # Build cage mesh (this may actually be the bottleneck...)
        t10 = time.time()
        v, f = gpytoolbox.lazy_cage(v_fine,f_fine,num_faces=4000,grid_size=100)
        t11 = time.time()
        cage_time = t11-t10
        if verbose:
            print("Built cage in",round(cage_time,3),"seconds.")
        # Tetrahedralize cage mesh
        t20 = time.time()
        tgen = tetgen.TetGen(v,f)
        nodes, elements =  tgen.tetrahedralize()
        t21 = time.time()
        tet_time = t21-t20
        if verbose:
            print("Tetrahedralization in ",round(tet_time,3),"seconds.")

        # Initialize fracture mode class
        t30 = time.time()
        modes = fracture_modes(nodes,elements) 
        # Set parameters for call to fracture modes
        params = fracture_modes_parameters(num_modes=num_modes,verbose=False,d=1)
        # Compute fracture modes. This should be the bottleneck:
        modes.compute_modes(parameters=params)
        modes.impact_precomputation(v_fine=v_fine,f_fine=f_fine)

        filename_without_extension = os.path.splitext(os.path.basename(filename))[0]
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        
        
        if compressed:
            modes.write_generic_data_compressed(output_dir)
            modes.write_segmented_modes_compressed(output_dir)
        else:
            modes.write_segmented_modes(output_dir,pieces=True)
        t31 = time.time()
        mode_time = t31-t30
        if verbose:
            print("Modes computed in ",round(mode_time,3),"seconds.")
        # # Generate random contact points on the surface
        B,FI = igl.random_points_on_mesh(100*num_impacts,v,f)
        B = np.vstack((B[:,0],B[:,0],B[:,0],B[:,1],B[:,1],B[:,1],B[:,2],B[:,2],B[:,2])).T
        P = B[:,0:3]*v[f[FI,0],:] + B[:,3:6]*v[f[FI,1],:] + B[:,6:9]*v[f[FI,2],:]
        sigmas = np.random.rand(100*num_impacts)*1000


        t40 = time.time()
        # Loop to generate many possible fractures
        running_num = 0
        for i in range(P.shape[0]):
            t400 = time.time()
            modes.impact_projection(contact_point=P[i,:],direction=np.array([1.0]),threshold=sigmas[i],num_modes_used=10)
            t401 = time.time()
            # if verbose:
            #     print("Impact simulation: ",round(t401-t400,3),"seconds.")
            if (modes.n_pieces_after_impact>1 and modes.n_pieces_after_impact<100):
                write_output_name = os.path.join(output_dir,"fractured_") +  str(running_num)
                running_num = running_num + 1
                if not os.path.exists(write_output_name):
                            os.mkdir(write_output_name)
                if compressed:
                    modes.write_segmented_output_compressed(filename=write_output_name)
                else:
                    modes.write_segmented_output(filename=write_output_name,pieces=True)
                t402 = time.time()
                # if verbose:
                #     print("Writing: ",round(t402-t401,3),"seconds.")
            if running_num >= num_impacts:
                break
        t41 = time.time()
        impact_time = t41-t40
        if verbose:
            print("Impacts computed in ",round(impact_time,3),"seconds.")
        t1 = time.time()
        total_time = t1-t0
        if verbose:
            print("Generated",running_num,"fractures for object",filename_without_extension,"and wrote them into",output_dir + "/","in",round(total_time,3),"seconds.")
    except:
        if verbose:
            print("Error encountered.")