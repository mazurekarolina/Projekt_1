from transformacje import Transformacje
import numpy as np

geo = Transformacje(model = input("Podaj model - wgs84, grs80 lub mars: "))
print("Pracuje nad tym...")
plik = "wsp_inp.txt"
# Odczyt z pliku: https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.genfromtxt.html
dane = np.genfromtxt(plik, delimiter=',', skip_header = 4)

# Obliczenie f, l, h
rows_flh = []
for x in range(dane.shape[0]):
    rows_flh.append(geo.xyz2plh(dane[x][0], dane[x][1], dane[x][2]))

dane = np.c_[dane, np.array(rows_flh)]

# Obliczenie N, E, U
rows_neu = []
for x in range(dane.shape[0]):
    rows_neu.append(geo.neu(dane[x][3], dane[x][4], dane[x][5], dane[x][0], dane[x][1], dane[x][2]))
dane = np.c_[dane, np.array(rows_neu)]

# Obliczenie u2000
rows_xy_2000 = []
for x in range(dane.shape[0]):
    rows_xy_2000.append(geo.u2000(dane[x][3], dane[x][4]))
dane = np.c_[dane, np.array(rows_xy_2000)]

# Obliczenie u1992
rows_xy_1992 = []
for x in range(dane.shape[0]):
    rows_xy_1992.append(geo.u2000(dane[x][3], dane[x][4]))
dane = np.c_[dane, np.array(rows_xy_1992)]

# Obliczenie kąta azymutu i kąta elewacji
rows_az_el = []
for x in range(dane.shape[0]):
    rows_az_el.append(geo.azymut_elewacja(dane[x][3], dane[x][4], dane[x][5], dane[x][0], dane[x][1], dane[x][2]))
dane = np.c_[dane, np.array(rows_az_el)]

# Odległości 2d oraz 3d
rows_2d_3d = []
for x in range(dane.shape[0]):
    try:
        rows_2d_3d.append([geo.odl2d(dane[x], dane[x+1]), geo.odl3d(dane[x], dane[x+1])])
    except IndexError:
        rows_2d_3d.append([0,0])

dane = np.c_[dane, np.array(rows_2d_3d)]

# Zapis: https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.savetxt.html
np.savetxt("wsp_out.txt", dane, delimiter=',  ', fmt = '%10.3f' , header = geo.header)

print("Sukces")
