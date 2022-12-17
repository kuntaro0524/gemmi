import sys
from matplotlib import pyplot
import gemmi

for path in sys.argv[1:]:
    mtz = gemmi.read_mtz_file(path)
    intensity = mtz.column_with_label('IMEAN')
    sigma = mtz.column_with_label('SIGIMEAN')
    if intensity is None or sigma is None:
        sys.exit("Columns I and SIGI not in the file: " + path)
    x = mtz.make_1_d2_array()
    y = intensity.array / sigma.array
    pyplot.figure(figsize=(5,5))
    pyplot.hexbin(x, y, gridsize=180, bins='log', cmap='gnuplot2')
    pyplot.show()
