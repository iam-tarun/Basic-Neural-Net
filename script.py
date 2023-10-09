f = open("./dataset/data_banknote_authentication.txt", "r")
lines = f.readlines()
newFile = ["v_wti,s_wti,c_wti,e_i,class\n"]
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
  data = line.split(",")
  # data[-1] = classMap[data[-1]]
  newFile.append(" ,".join(data))
f.close()
f1 = open("./dataset/bank_note.csv", "w")
f1.writelines(newFile)
f1.close()