# ... (previous code unchanged)

# Plot the results: average energy, ergotropy for all batteries, and max‐ergotropy line for battery 1
plt.figure(figsize=(10, 9))

for i in range(n_batt):
    plt.plot(collisions, avg_energy[i],
             label=f"Battery {i+1} Average Energy",
             linewidth=2)
    plt.plot(collisions, ergotropy[i],
             linestyle=':',
             label=f"Battery {i+1} Ergotropy",
             linewidth=2)

# vertical line at the collision of max ergotropy (for battery 1)
plt.rcParams["font.family"] = "Times New Roman"
plt.tick_params(axis='both', which='major', labelsize=18)
plt.axvline(x=collision_max,
            linestyle='--',
            label="Battery 1 Peak Ergotropy",
            color='red')

plt.xlabel("Number of Collisions", fontsize=18)
plt.ylabel("Average Energy and Ergotropy", fontsize=18)
plt.title("Possible Work Extraction for a 3-Level Battery system", fontsize=18)
plt.legend(fontsize=18)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
