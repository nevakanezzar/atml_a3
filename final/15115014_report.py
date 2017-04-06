import os

cats = ['success/', 'failure/', 'variance/']
text = ['Successful category', 'Failure category', 'High variance category', 'Ground truth', 'Mask']
dirs = [1, 10, 28, 300, 'gt', 'mask']

extension = ".png"

for i, cat in enumerate(cats):
    for direc in dirs:
        # directory = "models/images/selected/variance/300/"
        directory = 'models/images/selected/'+cat+str(direc)+'/'
        files = sorted([file for file in os.listdir(directory) if file.lower().endswith(extension)])

        images_in_row = 4

        if type(direc) == int:
            print r"\begin{figure}[!ht]"
            print r"\begin{center}"

            for num, file in enumerate(files):
                f2 = directory + file
                n = str.split(file,".")[0]
                print r"\subfigure{"
                print r"\includegraphics[width=2cm]{%s}" % f2
                print r"}"
                if (num+1)%images_in_row == 0:
                    print r"\\"

            print r"\caption{%s pixel predictions (%s) with cross entropy. Models arranged column-wise starting from single layer with 32 units, 3-layer with 32 units, single layer with 64 units, and single layer with 128 units.}" % (str(direc), text[i])
            print r"\end{center}"
            print r"\end{figure}"
        elif direc == 'gt':
            print r"\begin{figure}[!ht]"
            print r"\begin{center}"

            for num, file in enumerate(files):
                f2 = directory + file
                n = str.split(file,".")[0]
                if n == '1':
                    print r"\subfigure[%s pixel]{" % n
                else:
                    print r"\subfigure[%s pixels]{" % n
                print r"\includegraphics[width=2cm]{%s}" % f2
                print r"}"
                if (num+1)%images_in_row == 0:
                    print r"\\"

            print r"\caption{Ground truths (%s). Cross entropy is to be ignored here.}" % text[i]
            print r"\end{center}"
            print r"\end{figure}"

cats = ['one/', '2x2/']
text = ['one pixel', '2x2 image patch']
images_in_row = 3

for i, cat in enumerate(cats):
    direc = 'figures/inpainting/'+cat
    print r"\newpage"
    files = sorted([file for file in os.listdir(direc) if file.lower().endswith(extension)])
    print r"\begin{figure}[!ht]"
    print r"\begin{center}"

    for num,file in enumerate(files):
        f2 = direc+file
        print r"\subfigure{"
        print r"\includegraphics[width=4cm]{%s}" % f2
        print r"}"
        if (num+1)%images_in_row == 0:
            print r"\\"

    print r"\caption{Samples of Masked image, ground truth, and in-paintings for %s}" % text[i]
    print r"\end{center}"
    print r"\end{figure}"

