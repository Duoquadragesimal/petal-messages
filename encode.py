import pandas

wordList = pandas.read_csv("durhack_word_list.csv")["word"].to_list()

word = input("Enter word: ")
number = wordList.index(word)
print(number)
print(f" = {number:b}")