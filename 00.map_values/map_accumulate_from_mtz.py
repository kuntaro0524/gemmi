import sys
import numpy as np
import gemmi

# Function to read PDB file
code_list=[]

# Calculating map
map_mtz_name = sys.argv[1]
mtz=gemmi.read_mtz_file(map_mtz_name)

# Columns for map coefficient
option_str=sys.argv[2]
if option_str == "2fofc":
    amp="FWT"
    phase="PHWT"
elif option_str=="fofc":
    amp="DELFWT"
    phase="PHDELWT"
else:
    print("No options!")
    sys.exit()

grid = mtz.transform_f_phi_to_map(amp,phase,sample_rate=3)

# Reading PDB for map estimation
counter = 0
pdb_filename = sys.argv[3]
st = gemmi.read_structure(pdb_filename)

print("SIGMA=", np.std(grid))
# estimating sigma
sigma = np.std(grid)

ac=0.0
if st.cell.is_crystal():
    st.add_entity_types()
    for chain in st[0]:
        polymer = chain.get_polymer()
        if polymer:
            low_bounds = [float('+inf')] * 3
            high_bounds = [float('-inf')] * 3
            for residue in polymer:
                for atom in residue:
                    pos = st.cell.fractionalize(atom.pos)
                    x,y,z=pos
                    mapvalue=grid.interpolate_value(gemmi.Fractional(x,y,z)) / sigma
                    print(x,y,mapvalue)
                    ac+=mapvalue

print(ac)
