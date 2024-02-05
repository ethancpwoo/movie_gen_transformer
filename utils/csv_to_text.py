"""
Converts csv files to a text file used for data in model.
"""

import csv
import re


def main():

    with open('../src/wiki_movie_plots_deduped.csv', encoding='utf8', newline='') as data: 
        with open('../src/summaries.txt', 'w', encoding='utf8') as file:
            reader = csv.reader(data)
            for row in reader:
                print('writing... \n')

                file.write(f'Title: {row[1]} \n')
                file.write(f'Genre: {row[5]} \n')
                file.write(f'Description: \n')

                for word in row[7]:
                    if word != '\n':
                        file.write(word)
                file.write('\n' + '\n')

    print('done')

if __name__ == '__main__':
    main()