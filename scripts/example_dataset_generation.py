from context import fracture_utility as fracture
import sys
import os


# Read input mesh
filename = "data/bunny_oded.obj"
if len(sys.argv)>1:
    filename = sys.argv[1]

# Choose output directory
output_dir = os.path.splitext(os.path.basename(filename))[0]

# Call dataset generation
fracture.generate_fractures(filename,output_dir=output_dir,verbose=True,compressed=False,cage_size=5000,volume_constraint=0.00)