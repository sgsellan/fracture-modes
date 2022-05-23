from context import fracture_utility as fracture
import sys
import os


filename = "data/bunny_oded.obj"
if len(sys.argv)>1:
    filename = sys.argv[1]

output_dir = os.path.splitext(os.path.basename(filename))[0]

fracture.generate_fractures(filename,output_dir=output_dir,verbose=False)