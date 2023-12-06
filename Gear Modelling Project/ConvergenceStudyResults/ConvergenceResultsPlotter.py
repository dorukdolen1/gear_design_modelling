import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# TODO: Open all files and parse fine and coarse mesh sizes
# TODO: Plot CPRESS vs spatial co-ordinate for different roll angles, one element size
# TODO: Plot max(CPRESS) vs roll angle, for different element sizes
# TODO: Cut tip stress out, and then plot max stress vs fine element size, for different element size


def main():

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

    cpress_study = []
    fine_size_cpress = []
    mises_study = []
    fine_size_mises = []

    for file in os.listdir():
        if file.split(".")[-1] == 'csv':
            fine = file.split("_")[2]
            coarse = file.split("_")[3]
            fine = fine.replace("0", "0.", 1)
            coarse = coarse.replace("0", ".", 1)
            if file.split("_")[-1].split(".")[0] == 'CPRESS':
                df = pd.read_csv(file)
                df.drop([0], inplace=True)
                roll_angle = df.columns[1:].astype('float')

                array = df.values[:, 1:].astype('float')

                cpress = np.amax(array, axis=0) / 1000 / 30

                ax[0, 0].plot(roll_angle, cpress, label=fine)

                n = len(cpress)
                
                cpress_study += [np.max(cpress[int(9/16*n):int(13/16*n)])]
                
                fine_size_cpress += [float(fine)]

            if file.split("_")[-1].split(".")[0] == 'MISES':
                df = pd.read_csv(file)
                df.drop([0], inplace=True)
                roll_angle = df.columns[1:].astype('float')

                array = df.values[:, 1:].astype('float')

                mises = np.amax(array, axis=0)

                ax[1, 0].plot(roll_angle, mises, label=fine)

                max_mises = max(mises[:int(0.9*len(mises))])

                mises_study += [max_mises]
                fine_size_mises += [float(fine)]

    T = 2500  # Nm
    b = 30  # mm
    m = 4.5
    z = 35
    omega = 1  # deg/s
    omega = np.radians(omega)  # rad/s
    r = m * z / 2 / 1000  # m

    F = T / r  # N
    v = omega * r  # m/s
    K = 6.1 / (6.1 + v)
    Y = 3.4

    stress = 2 * T * Y / (m * b * m * z) * 1000  # MPa

    db1 = 148.00159
    db2 = 232.57392
    alpha_tw = 0.3821467  # Radians

    R1 = db1 * np.tan(alpha_tw) / 2
    R2 = db2 * np.tan(alpha_tw) / 2

    R = (1/R1 + 1/R2)**-1  # mm

    E = 207e9  # Pa
    
    Es = (2 * (1 - 0.3**2) / E)**-1

    P = 2 * T / (b * db1) * 1e6  # N

    p_max = np.sqrt(P * Es / (np.pi * R*1e-3))

    ax[0, 0].set_title("Maximum CPRESS vs Roll Angle")
    ax[0, 0].set_xlim([4, None])
    ax[0, 0].set_ylim([0, None])
    ax[0, 0].set_xlabel("Roll Angle [deg]")
    ax[0, 0].set_ylabel("Contact Pressure [GPa]")
    ax[0, 0].legend(title="Fine Element Size [mm]")

    ax[0, 1].plot(fine_size_cpress, cpress_study, label="FEA Pressure")
    ax[0, 1].plot([min(fine_size_mises), max(fine_size_mises)], [p_max*1e-9, p_max*1e-9], linestyle='--', color='red',
                  label="Hertz Equation Contact Pressure")
    ax[0, 1].set_xscale('log')
    ax[0, 1].set_title("CPRESS at central location vs Fine Mesh Size")
    ax[0, 1].set_xlabel("Fine Mesh Size [mm]")
    ax[0, 1].set_ylabel("Contact Pressure [GPa]")
    ax[0, 1].legend()

    ax[1, 0].set_title("Maximum Root MISES Stress vs Roll Angle")
    ax[1, 0].set_xlim([4, None])
    ax[1, 0].set_ylim([0, None])
    ax[1, 0].set_xlabel("Roll Angle [deg]")
    ax[1, 0].set_ylabel("Root Stress [MPa]")
    ax[1, 0].legend(title="Fine Element Size [mm]")

    ax[1, 1].plot(fine_size_mises, mises_study, label="FEA Stress")
    ax[1, 1].set_xscale('log')
    ax[1, 1].plot([min(fine_size_mises), max(fine_size_mises)], [stress, stress], linestyle='--', color='red',
                  label="Lewis Equation Stress")
    ax[1, 1].set_title("Maximum Root Stress vs Fine Mesh Size")
    ax[1, 1].set_xlabel("Fine Mesh Size [mm]")
    ax[1, 1].set_ylabel("Root Stress Pressure [MPa]")
    ax[1, 1].legend()

    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    main()
