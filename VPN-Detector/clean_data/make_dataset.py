import pandas as pd
import os

root = "clean_data"
vpn  = pd.read_csv(os.path.join(root, "vpn15s_cleaned.csv"))
nov  = pd.read_csv(os.path.join(root, "novpn15s_cleaned.csv"))

vpn ["label"] = 1      # 1 = VPN
nov ["label"] = 0      # 0 = Non‑VPN

df = pd.concat([vpn, nov], ignore_index=True)
df.to_csv(os.path.join(root, "vpn15s_labeled.csv"), index=False)
print("✅ Saved clean_data/vpn15s_labeled.csv  →  rows:", len(df))
