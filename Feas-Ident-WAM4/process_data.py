import numpy as np

# File paths
names = ['slax/rbtlog_2',
    'slax/rbtlog_3',
    'slax/rbtlog_9',
    'slax/rbtlog_10',
    'slax/rbtlog_11',]

for name in names:
                    
    input_file = f"data/recdata/{name}.dat"
    output_file = f"data/recdata/{name}_4_dof.dat"

    # Load full data: shape (N, 15)
    # [time, q1...q7, tau1...tau7]
    data = np.loadtxt(input_file)

    # Extract columns
    time = data[:, [0]]            # shape (N, 1)
    positions = data[:, 1:5]       # q1 to q4
    torques = data[:, 8:12]        # tau1 to tau4

    # Concatenate: [time | q1-q4 | tau1-tau4]
    output_data = np.hstack((time, positions, torques))

    # Save
    np.savetxt(output_file, output_data, fmt="%.10f")

    print(f"Saved 4-DoF data to '{output_file}' with shape {output_data.shape}")
