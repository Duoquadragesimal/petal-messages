import pandas

wordList = pandas.read_csv("durhack_word_list.csv")["word"].to_list()

binary = input("Enter binary: ")
number = int(binary, 2)
print(wordList[number])