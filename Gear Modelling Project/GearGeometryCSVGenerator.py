import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


def involute(alpha, r_b):
    r = r_b / np.cos(alpha)
    phi = np.tan(alpha) - alpha

    return r, phi


def inv(angle):
    return np.tan(np.radians(angle)) - np.radians(angle)


def main():
    CID = "01739141"
    d1 = int(CID[0])
    d2 = int(CID[1])
    d3 = int(CID[2])
    d4 = int(CID[3])
    d5 = int(CID[4])
    d6 = int(CID[5])
    d7 = int(CID[6])
    d8 = int(CID[7])

    # Personalised gear tooth geometry
    mn = 4 + 0.5 * d8  # Module
    z1 = 25 + d5 + d6  # Pinion tooth number
    x1 = 0.1 * (d1 + d2)  # Pinion addendum correction factor
    z2 = 50 + d7 + d8  # Wheel tooth number
    x2 = 0.05 * (d3 + d4)  # Wheel addendum correction factor

    sum_x = x1 + x2
    G = z2 / z1  # Gear ratio

    # Tooth parameters
    alpha_n = np.radians(20)  # Pressure angle
    beta = np.radians(0)
    # a = 0.5 * mn * (z1 + z2) + 0.5 * sum_x + 3  # Unmodified tight centre distance

    def func(x):
        return (np.tan(x) - x - np.tan(alpha_n) + alpha_n) / np.tan(alpha_n) - (x1 + x2) / (0.5 * (z1 + z2))

    alpha_tw = fsolve(func, 0.38)[0]
    a = mn * np.cos(alpha_n) * 0.5 * (z1+z2) / np.cos(alpha_tw)

    # Calculating reference diameters for gear 1
    d_ref_1 = z1 * mn
    da_1 = d_ref_1 + 2 * mn * (x1 + 1)
    df_1 = d_ref_1 + 2 * mn * (x1 - 1.25)
    db_1 = d_ref_1 * np.cos(alpha_n)

    # Calculating reference diameters for gear 2
    d_ref_2 = z2 * mn
    da_2 = d_ref_2 + 2 * mn * (x2 + 1)
    df_2 = d_ref_2 + 2 * mn * (x2 - 1.25)
    db_2 = d_ref_2 * np.cos(alpha_n)

    # Working pressure angle and Path of Contact
    # alpha_tw = np.arccos((db_1 + db_2) / (2 * a))
    g_alpha = np.sqrt((da_1 / 2) ** 2 - (db_1 / 2) ** 2) + np.sqrt((da_2 / 2) ** 2 - (db_2 / 2) ** 2) - (db_1 + db_2) \
              * np.tan(alpha_tw) / 2

    # Epsilon_alpha check (must be between 1 and 2, preferably around 1.7)
    epsilon_alpha = g_alpha / (np.pi * mn * (1/np.cos(beta)) * np.cos(alpha_n))

    # Addition Defined Parameters for GearGeometry.csv
    dw_1 = 130  # Pinion web outer diameter
    ds_1 = 100  # Pinion shaft outer diameter
    R1 = 1.5  # Pinion root fillet radius
    n1 = 5  # Number of modelled teeth

    dw_2 = 220  # Wheel web outer diameter
    ds_2 = 200  # Wheel shaft outer diameter
    R2 = 1.5  # Wheel root fillet radius
    n2 = 5  # Number of modelled teeth

    # Max Hertz Pressure
    T = 2500  # Pinion torque, Nm
    b = 0.03  # Facewidth, m
    E = 207e9  # Elastic Modulus, Pa
    v = 0.3  # Poission Ratio
    t_w = 20  # Web Thickness
    delta2 = (1 + 1/G) * (alpha_tw + inv(alpha_n)) + 2*(x1+x2)*np.tan(alpha_n) / z2 - 2*a*np.sin(alpha_tw) / (z2 * mn * np.cos(alpha_n))

    # Printing Gear Geometry Values
    print("d_f1,", df_1, sep='')
    print("d_w1,", dw_1, sep='')
    print("d_s1,", ds_1, sep='')
    print("z1,", z1, sep='')
    print("R1,", R1, sep='')
    print("n1,", n1, sep='')
    print("", sep='')
    print("d_f2,", df_2, sep='')
    print("d_w2,", dw_2, sep='')
    print("d_s2,", ds_2, sep='')
    print("z2,", z2, sep='')
    print("R2,", R2, sep='')
    print("n2,", n2, sep='')
    print("", sep='')
    print("a,", round(a, 0), sep='')
    print("T,", T*1000, sep='')
    print("b,", b*1000, sep='')
    print("t_w,", t_w, sep='')
    print("delta2,", round(delta2, 2), sep='')
    print("gamma2,", -5.45, sep='')
    print("", sep='')

    # CHANGE THIS TO NOT OVERWRITE PREVIOUS JOB
    job_name = "job_1"

    # VARY THESE PARAMETERS FOR RUNNING THE PYTHON #
    print("webElementSize,", 2, sep='')
    print("coarseElementSize,", 5, sep='')
    print("fineElementSize,", 1, sep='')
    print("refineMeshAtTooth,", 2, sep='')
    print("numTeethRefined,", 3, sep='')
    print("exportDataAtTooth,", 3, sep='')
    print("numTeethExported,", 1, sep='')
    print("rotateTimeIncrement,", 0.4, sep='')
    print("rotateTime,", 16, sep='')
    print("runCoarseMesh?,", 0, sep='')
    print("runRefinedMesh?,", 1, sep='')
    print("jobName,", job_name, sep='')
    print("")
    # VARY THESE PARAMETERS FOR RUNNING THE PYTHON #

    print("N,", 10, sep='')

    # Parametization of g_alpha
    n = 10  # Number of points to parametize
    xi = np.linspace(0, g_alpha, n)  # Parameter

    # GEAR 1 PLOT
    t1a = np.sqrt((da_1/2)**2 - (db_1/2)**2)
    t1b = t1a - g_alpha
    t1_xi = t1b + xi

    alpha_1 = np.arctan(t1_xi / (db_1/2))

    r_1, phi_1 = involute(alpha_1, (db_1/2))

    s1_ref = mn * (np.pi/2 + 2*x1*np.tan(alpha_n))
    s1_b = db_1 * (s1_ref/d_ref_1 + np.tan(alpha_n) - alpha_n)

    phi_1_i_rot = np.pi / 2 + s1_b/db_1 - phi_1

    x1 = np.multiply(r_1, np.cos(phi_1_i_rot))
    y1 = np.multiply(r_1, np.sin(phi_1_i_rot))

    # GEAR 1 PLOT
    t2a = np.sqrt((da_2 / 2) ** 2 - (db_2 / 2) ** 2)
    t2b = t2a - g_alpha
    t2_xi = t2b + xi

    alpha_2 = np.arctan(t2_xi / (db_2 / 2))

    r_2, phi_2 = involute(alpha_2, (db_2 / 2))

    s2_ref = mn * (np.pi / 2 + 2 * x2 * np.tan(alpha_n))
    s2_b = db_2 * (s2_ref / d_ref_2 + np.tan(alpha_n) - alpha_n)

    phi_2_i_rot = np.pi / 2 + s2_b / db_2 - phi_2

    x2 = np.multiply(r_2, np.cos(phi_2_i_rot))
    y2 = np.multiply(r_2, np.sin(phi_2_i_rot))

    P_nmax = T / (b * (db_1/2)) * 1e3
    E_star = ((1-v**2)/E + (1-v**2)/E)**-1

    P_n_1 = []
    theta_1 = t1_xi / (db_1 / 2)
    theta_2 = t2_xi / (db_2 / 2)

    R_i_1 = (1/(0.5*db_1*theta_1) + 1/(0.5*db_2*theta_2))**-1 * 1e-3

    for x in xi:
        if 0 <= x/g_alpha < 1 - 1/epsilon_alpha:
            P_n_1 += [P_nmax * (x/g_alpha) / (1 - 1/epsilon_alpha)]
        elif 1 - 1/epsilon_alpha <= x / g_alpha < 1/epsilon_alpha:
            P_n_1 += [P_nmax]
        elif 1/epsilon_alpha <= x / g_alpha <= 1:
            P_n_1 += [P_nmax * (1 - (x/g_alpha - 1/epsilon_alpha)/(1 - 1/epsilon_alpha))]

    P_0_1 = np.sqrt(np.divide(E_star * np.array(P_n_1), np.pi * R_i_1))

    sigma_b = 2 * T * 3.4 / (d_ref_1 * b * mn)

    print("flankx1,flanky1")
    for row in np.column_stack([x1, y1]):
        print(np.round(row[0], 3), ",", np.round(row[1], 3), sep='')

    print("")

    print("flankx2,flanky2")
    for row in np.column_stack([x2, y2]):
        print(np.round(row[0], 3), ",", np.round(row[1], 3), sep='')


if __name__ == "__main__":
    main()
