import numpy as np
import matplotlib.pyplot as plt

def rotation_matrix(theta_rad):
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])

def translation_matrix(tx, ty):
    return np.array([[1, 0, tx],
                     [0, 1, ty],
                     [0, 0, 1]])

def forward_kinematics(L1, L2, theta1_deg, theta2_deg):
    theta1_rad = np.radians(theta1_deg)
    theta2_rad = np.radians(theta2_deg)
    
    H1_0 = rotation_matrix(theta1_rad)
    H2_1 = translation_matrix(L1, 0)
    H3_2 = rotation_matrix(theta2_rad)
    H4_3 = translation_matrix(L2, 0)
    H4_0 = H1_0 @ H2_1 @ H3_2 @ H4_3

    #Posisi (x, y) ada di kolom terakhir matriks
    x = H4_0[0, 2]
    y = H4_0[1, 2]

    #Untuk plotting
    H_j1 = H1_0 @ H2_1
    p_j1_x = H_j1[0, 2]
    p_j1_y = H_j1[1, 2]

    return (x, y), (p_j1_x, p_j1_y)

def inverse_kinematics(L1, L2, x, y):
    D = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    theta2_rad = np.arccos(D)

    k1 = L1 + L2 * np.cos(theta2_rad)
    k2 = L2 * np.sin(theta2_rad)
    theta1_rad = np.arctan2(y, x) - np.arctan2(k2, k1)

    theta1_deg = np.degrees(theta1_rad)
    theta2_deg = np.degrees(theta2_rad)
    
    return theta1_deg, theta2_deg

def visualisasi_arm(L1, L2, theta1_deg, theta2_deg, p_j1_x, p_j1_y, x_end, y_end):

    plt.figure()

    # Plot links arm
    plt.plot([0, p_j1_x], [0, p_j1_y], 'r-o', linewidth=3, markersize=10, label=f'Femur (L1={L1})')
    plt.plot([p_j1_x, x_end], [p_j1_y, y_end], 'g-o', linewidth=3, markersize=10, label=f'Tibia (L2={L2})')
    
    # Plot sendi
    plt.plot(0, 0, 'k.', markersize=20, label='Sendi Coxa (Origin)')
    plt.plot(x_end, y_end, 'b*', markersize=15, label=f'End-Effector ({x_end:.2f}, {y_end:.2f})')
    
    plt.title(f'Visualisasi 2-DoF Arm\nθ1={theta1_deg:.2f}°, θ2={theta2_deg:.2f}°')
    plt.xlabel('Koordinat X')
    plt.ylabel('Koordinat Y')
    plt.legend()
    plt.grid(True)
    
    # Set axis agar proporsional
    max_range = (L1 + L2) * 1.2
    plt.axis('equal')
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)

    plt.savefig('hasil_visualisasi.png')
    plt.show()

if __name__ == "__main__":
    L1 = float(input("Masukkan panjang link femur (L1) 2 digit terakhir NIU: "))
    L2 = float(input("Masukkan panjang link tibia (L2) 2 digit terakhir NIF: "))

    mode = input("Pilih mode (1: FK, 2: IK): ")

    if mode == '1':
        theta1_deg = 40.0
        theta2_deg = 30.0
        
        (x_end, y_end), (p_j1_x, p_j1_y) = forward_kinematics(L1, L2, theta1_deg, theta2_deg)
        print(f"Posisi End-Effector: x = {x_end:.2f}, y = {y_end:.2f}")
        
        visualisasi_arm(L1, L2, theta1_deg, theta2_deg, p_j1_x, p_j1_y, x_end, y_end)

        print(f"posisi x: {x_end:.2f}")
        print(f"posisi y: {y_end:.2f}")

        visualisasi_arm(L1, L2, theta1_deg, theta2_deg, p_j1_x, p_j1_y, x_end, y_end)

    elif mode == '2':
        x = float(input("Masukkan posisi x end-effector: "))
        y = float(input("Masukkan posisi y end-effector: "))
        
        theta1_deg, theta2_deg = inverse_kinematics(L1, L2, x, y)
        print(f"Sudut yang dibutuhkan: θ1 = {theta1_deg:.2f}°, θ2 = {theta2_deg:.2f}°")
        
        (x_end, y_end), (p_j1_x, p_j1_y) = forward_kinematics(L1, L2, theta1_deg, theta2_deg)
        
        visualisasi_arm(L1, L2, theta1_deg, theta2_deg, p_j1_x, p_j1_y, x_end, y_end)