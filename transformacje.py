from math import sin, cos, sqrt, atan, atan2, degrees, radians, pi, tan, asin
from pickletools import float8
import numpy as np

class Transformacje:
    def __init__(self, model: str = "wgs84"):
        """
        Parametry elipsoid:
            a - duża półoś elipsoidy - promień równikowy
            b - mała półoś elipsoidy - promień południkowy
            flat - spłaszczenie
            ecc2 - mimośród^2
        + WGS84: https://en.wikipedia.org/wiki/World_Geodetic_System#WGS84
        + Inne powierzchnie odniesienia: https://en.wikibooks.org/wiki/PROJ.4#Spheroid
        + Parametry planet: https://nssdc.gsfc.nasa.gov/planetary/factsheet/index.html
        """
        if model == "wgs84":
            self.A = 6378137.0 # semimajor_axis
            self.B = 6356752.31424518 # semiminor_axis
        elif model == "grs80":
            self.A = 6378137.0
            self.B = 6356752.31414036
        elif model == "mars":
            self.A = 3396900.0
            self.B = 3376097.80585952
        else:
            raise NotImplementedError(f"{model} model not implemented")
        self.flat = (self.A - self.B) / self.A
        self.ecc = sqrt(2 * self.flat - self.flat ** 2) # eccentricity  WGS84:0.0818191910428 
        self.ecc2 = (2 * self.flat - self.flat ** 2) # eccentricity**2
        print(self.ecc2)


    
    def xyz2plh(self, X, Y, Z, output = 'dec_degree'):
        """
        Algorytm Hirvonena - algorytm transformacji współrzędnych ortokartezjańskich (x, y, z)
        na współrzędne geodezyjne długość szerokość i wysokośc elipsoidalna (phi, lam, h). Jest to proces iteracyjny. 
        W wyniku 3-4-krotneej iteracji wyznaczenia wsp. phi można przeliczyć współrzędne z dokładnoscią ok 1 cm.     
        Parameters
        ----------
        X, Y, Z : FLOAT
             współrzędne w układzie orto-kartezjańskim, 

        Returns
        -------
        lat
            [stopnie dziesiętne] - szerokość geodezyjna
        lon
            [stopnie dziesiętne] - długośc geodezyjna.
        h : TYPE
            [metry] - wysokość elipsoidalna
        output [STR] - optional, defoulf 
            dec_degree - decimal degree
            dms - degree, minutes, sec
        """
        r   = sqrt(X**2 + Y**2)           # promień
        lat_prev = atan(Z / (r * (1 - self.ecc2)))    # pierwsze przybliilizenie
        lat = 0
        while abs(lat_prev - lat) > 0.000001/206265:    
            lat_prev = lat
            N = self.A / sqrt(1 - self.ecc2 * sin(lat_prev)**2)
            h = r / cos(lat_prev) - N
            lat = atan((Z/r) * (((1 - self.ecc2 * N/(N + h))**(-1))))
        lon = atan(Y/X)
        N = self.A / sqrt(1 - self.ecc2 * (sin(lat))**2);
        h = r / cos(lat) - N       
        if output == "dec_degree":
            return degrees(lat), degrees(lon), h 
        elif output == "dms":
            lat = self.deg2dms(degrees(lat))
            lon = self.deg2dms(degrees(lon))
            return f"{lat[0]:02d}:{lat[1]:02d}:{lat[2]:.2f}", f"{lon[0]:02d}:{lon[1]:02d}:{lon[2]:.2f}", f"{h:.3f}"
        else:
            raise NotImplementedError(f"{output} - output format not defined")

    def fl2xy(self, ff: float, ll: float) -> tuple:
        f = ff*pi/180
        l = ll*pi/180
        b2 = (self.A**2)*(1-self.ecc2)
        ep2 = (self.A**2-b2)/b2
        t = tan(f)
        n2 = ep2*(cos(f)**2)
        N = self.func_n(f)
        si = self.sigma(f)
        dL = l - self.get_l0(ll) 
        xgk = si + ((dL**2)/2)*N*sin(f)*cos(f)*(1 + (dL**2/12)*cos(f)**2*(5 - t**2 + 9*n2 + 4*n2**2) + (dL**4/360)*cos(f)**4*(61 - 58*t**2 + t**4 + 270*n2 - 330*n2*t**2))
        ygk = dL*N*cos(f)*(1 + (dL**2/6)*cos(f)**2*(1 - t**2 + n2) + (dL**4/120)*cos(f)**4*(5 - 18*t**2 + t**4 + 14*n2 - 58*n2*t**2))
        return(xgk,ygk)

    def u2000(self, f:float, l:float):
        m2000 = 0.999923
        xgk, ygk = self.fl2xy(f, l)
        l0 = self.get_l0(l)
        x = xgk * m2000
        y = ygk * m2000 + (l0*180/pi/3)* 1000000 + 500000;
        return(x,y)

    def sigma (self, f: float) -> float:
        """
        Metoda przyjmuje fi wyliczone z hirvonena i oblicza sigmę
        """
        A0 = 1 - (self.ecc2/4)-((3*(self.ecc2**2))/64)-((5*(self.ecc2**3))/256)
        A2 = (3/8)*(self.ecc2+((self.ecc2**2)/4)+((15*(self.ecc2**3))/128))
        A4 = (15/256)*((self.ecc2**2)+((3*(self.ecc2**3))/4))
        A6 = (35*(self.ecc2**3))/3072
        return self.A*(A0*f - A2*sin(2*f) + A4*sin(4*f) - A6*sin(6*f))

    def get_l0(self, l: float) -> int:
        """
        Metoda wyznacza właścwe L0 na podstawie l
        """
        if 13.5 < l <= 16.5:
            return 15*pi/180
        if 16.5 < l <= 19.5:
            return 18*pi/180
        if 19.5 < l <= 22.5:
            return 21*pi/180
        if 22.5 < l <= 25.5:
            return 24*pi/180

    def u1992(self, xgk:float, ygk:float):
        x = xgk * 0.9993-5300000
        y = ygk *0.9993+500000
        return x, y

    def func_n(self, f: float) -> float:
        """
        Metoda licząca promień krzywizny w I wertykale
        """
        N = (self.A)/sqrt(1-self.ecc2*(sin(f)**2))
        return N

    def flh2xyz(self, f:float, l:float, h:float):
        N = self.func_n(f)
        x = (N + h)*cos(f)*cos(l)
        y = (N + h)*cos(f)*sin(l)
        z = (N*(1-self.ecc2)+h)*sin(f)
        return x, y, z

    def azymut_elewacja(self, f, l, h):

        n, e, u = self.neu(f, l, h)

        #AZYMUT I ELEWACJA DO SATELITY

        azymut = atan2(e, n)
        azymut = np.rad2deg(azymut)
        azymut = azymut + 360 if azymut < 0 else azymut

        elewacja = asin(u/(sqrt(e**2+n**2+u**2)))
        elewacja = np.rad2deg(elewacja)

        return azymut, elewacja

    def neu(self, f, l, h):
        N = self.func_n(f)
        
        Xp = (N + h) * cos(f) * cos(l)
        Yp = (N + h) * cos(f) * sin(l)
        Zp = (N * (1 - self.ecc2) + h) * sin(f)
        
        XYZp = np.array([Xp, Yp, Zp])
        XYZs = np.array([X, Y, Z])
        
        XYZ = XYZs - XYZp
        XYZ = np.array(XYZ)
        	
        Rneu = np.array([[-sin(f)*cos(l), -sin(l), cos(f)*cos(l)],
                         [-sin(f)*sin(l), cos(l), cos(f)*sin(l)],
                         [cos(f), 0, sin(f)]])
        
        n, e, u = Rneu.T @ XYZ
        return n, e, u

    def odl2d(self, p1, p2):
        x1, y1 = p1[0], p1[1]
        x2, y2 = p2[0], p2[1]
        return sqrt((x2 - x1)**2+(y2 - y1)**2)

    def odl2d(self, p1, p2):
        x1, y1, z1 = p1[0], p1[1], p1[2]
        x2, y2, z2 = p2[0], p2[1], p1[2]
        return sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)




if __name__ == "__main__":
    # utworzenie obiektu
    """
    geo = Transformacje(model = "wgs84")
    # dane XYZ geocentryczne
    X = 3664940.500; Y = 1409153.590; Z = 5009571.170
    phi, lam, h = geo.xyz2plh(X, Y, Z)
    print(phi, lam, h)
     """   
    geo = Transformacje("wgs84")
    f, l, h = geo.xyz2plh(X = 3664940.500, Y = 1409153.590, Z = 5009571.170)
    print(f,l)
    print(geo.fl2xy(f,l))
    
