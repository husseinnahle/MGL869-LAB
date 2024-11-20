import os, shutil

# ******** à modifier ******** #
src = src = "C:\\Users\\lafor\\Desktop\\ETS - Cours\\MGL869-01_Sujets spéciaux\\Laboratoire\\Hive\\hive"
dst = "C:\\Users\\lafor\\Desktop\\ETS - Cours\\MGL869-01_Sujets spéciaux\\Laboratoire\\HiveJavaFiles"
# **************************** #

for root, dirs, files in os.walk(src):
    for file in files:
        if file.endswith(".java"):
            shutil.copy2(os.path.join(root, file), dst)