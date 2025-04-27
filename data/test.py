# ---------------------------------------------
# 🔧 NOTE FOR FUTURE DEVELOPERS:
# The block below is for **testing purposes only**.
# It will only run when this script is executed directly,
# and will NOT run if this file is imported as a module.
# ---------------------------------------------
# Sample loop to check one training batch
for batch in train_loader:
    input_ids = batch['input_ids']
    print(f"✅ Sample batch shape: {input_ids.shape}")
    break

