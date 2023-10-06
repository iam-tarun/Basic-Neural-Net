f = open("./yeast/yeast.data", "r")
lines = f.readlines()
newFile = ["mcg,gvh,alm,mit,erl,pox,vac,nuc,localization_site\n"]
classMap = {
  "CYT\n": "0\n",
  "NUC\n": "1\n",
  "MIT\n": "2\n",
  "ME3\n": "3\n",
  "ME2\n": "4\n",
  "ME1\n": "5\n",
  "EXC\n": "6\n",
  "VAC\n": "7\n",
  "POX\n": "8\n",
  "ERL\n": "9\n"
}
for line in lines:
  data = line.split("  ")
  data[-1] = classMap[data[-1]]
  newFile.append(" ,".join(data[1::]))
f.close()
f1 = open("./yeast/yeast.csv", "w")
f1.writelines(newFile)
f1.close()