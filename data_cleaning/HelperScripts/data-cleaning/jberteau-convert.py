#! /bin/python3

import sys
import os

# Python's split(",") method isn't enough
# as that method doesn't respect deliminators in quoted substrings
# NOTE: This will only treat commas NOT between a pair of quotes "" as deliminators
# NOTE: This does NOT accout for escaped quotes in the file (e.g., \"\")

def tokenize(line):
    quoted = False
    token = ""
    tokens = []
    for char in line:
        if char == "\"":
            quoted = not quoted
            continue
        if char == "," and not quoted:
            tokens += [token]
            token = ""
            continue
        token += char

    # if there was a trailing token not followed by a final comma, append it
    if token:
        tokens += [token]

    return tokens

class CsvColumn:
    def __init__(self):
        self._index = None
        self._value = ""

    def set_index(self, index):
        self._index = index

    def get_index(self):
        return self._index

    def set_value(self, value):
        self._value = value

    def get_value(self):
        return self._value

    def get_tokens(self):
        return [self._value]

#Energy level column 
class ELevel(CsvColumn):
    def __init__(self):
        super().__init__()
        self._energy = None
        self._uncertainty = None

    def get_tokens(self):
        if self._value is None:
            raise Exception("Member value is None")

        tokens = self._value.split(" ")
        if len(tokens) < 2:
            raise Exception("String \"{}\" split in fewer than 2 tokens".format(self._value))

        return tokens

#JPi column
class JPi(CsvColumn):
    def __init__(self):
        super().__init__()

    def get_tokens(self):
        if self._value is None:
            raise Exception("Member value is None")

        tokens = self._value.split(" ")
        if len(tokens) < 1:
            raise Exception("String \"{}\" split in fewer than 1 token".format(self._value))

        prunes = ("(", ")") # Remove grouping characters
        for prune in prunes:
            tokens[0] = tokens[0].replace(prune, "")

        if not tokens[0]:
            raise Exception("String is empty")

        if "," in tokens[0]:
            raise Exception("String \"{}\" contains deliminator, multiple values will be skipped".format(self._value))

        if not ("+" in tokens[0] or "-" in tokens[0]):
            raise Exception("String \"{}\" does not contain \"+\" or \"-\"".format(self._value))

        return tokens

#Converts csv to ASCII readable file (Ags)
class Converter:
    def __init__(self, csv_file_name, ags_file_name):
        self._debug = True

        self._csv_file_name = csv_file_name
        self._csv_file = None

        self._ags_file_name = ags_file_name
        self._ags_file = None

        self._find_indexes = True
        self._csv_counter = 0
        self._ags_counter = 0

        self._e_level = ELevel()
        self._jpi_level = JPi()

        self._column_dict = {
            "E(level)(keV)" : self._e_level,
            "Jπ(level)" : self._jpi_level,
        }

    
        self._open_csv_file()
        self._open_ags_file()
        for line in self._csv_file:

            # There are some characters which impede parsing
            # I'm resorting to manually pruning them
            prunes = ("?", "\xa0")
            for prune in prunes:
                line = line.replace(prune, "")

            try:
                self._loop(line)
            except Exception as e:
                print(e)
                break

        if self._csv_file is not None:
            self._csv_file.close()
        if self._ags_file is not None:
            self._ags_file.close()

    def _open_csv_file(self):
        # Do not try except here
        # Exceptions raised by open should cause the program to fail
        # the specification '-sig' below handles the byte-order-mark at the beginning of the file
        self._csv_file = open(self._csv_file_name, 'r', encoding='utf-8-sig')
        self._csv_counter = 0

    def _open_ags_file(self):
        # Do not try except here
        # Exceptions raised by open should cause the program to fail
        self._ags_file = open(self._ags_file_name, 'w', encoding='utf-8')
        self._ags_file.write("** ASCII Graphical Level Scheme file.\n")
        self._ags_file.write("** First five lines are reserved for comments.\n")
        self._ags_file.write("** This file created 03-May-00 14:47:38\n")
        self._ags_file.write("** Program GLS,  author A.Rounds \n")
        self._ags_file.write("**\n")
        self._ags_file.write(" ASCII GLS file format version 1.0\n")
        self._ags_file.write("**  Z Nlevels Ngammas  Nbands Nlabels CharSizeX CharSizeY ArrowWidFact\n")
        self._ags_file.write("   00     000     000      00       0     75.00     85.00      3.00\n")
        self._ags_file.write("**  MaxArrowTan MaxArrowDX DefBandWid DefBandSep ArrowWidth ArrowLength\n")
        self._ags_file.write("       0.267900     999.00     600.00     150.00      40.00      80.00\n")
        self._ags_file.write("** ArrowBreak LevelBreak   LevCSX   LevCSY LevEnCSX LevEnCSY   GamCSX   GamCSY\n")
        self._ags_file.write("       30.00       40.00    75.00    85.00    75.00    85.00    75.00    85.00\n")
        self._ags_file.write("** Level   Energy +/- err     Jpi     K Band# LevelFlag LabelFlag EnLabFlag\n")
        self._ags_file.write("++   LabelDX   LabelDY EnLabelDX EnLabelDY  LevelDX1  LevelDX2\n")
        self._ags_counter = 0

    def _loop(self, line):
        self._csv_counter += 1

        if line.isspace():
            # A blank line marks a re-declaration of column headers
            self._find_indexes = True
            return

        line = line[:-1] # Remove trailing newline character
        tokens = tokenize(line)

        # If we need to identify positions of columns of interest, do that
        if self._find_indexes:
            self._find_header_indexes(tokens)
            return

        # Otherwise, associate the tokens with a value, positionally
        for column in self._column_dict.values():
            if len(tokens) < column.get_index():
                raise Exception("{}:{:n} Expected at least {:n} tokens".format(self._csv_file_name, self._csv_counter, column.get_index()))
            column.set_value(tokens[column.get_index()])

        if self._debug:
            print()
            print("{}:{:n}".format(self._csv_file_name, self._csv_counter))
            for name, column in self._column_dict.items():
                print("\t{} has token \"{}\"".format(name, column.get_value()))

        self._append_tokens()

    def _find_header_indexes(self, tokens):
        # 'tokens' is an ordered list of the names of the column headers obtained from a successful parse
        # Create a dictionary of column headers to positions
        position_dict = {}
        for i in range(len(tokens)): # Must do this since positional index is important
            position_dict[tokens[i]] = i
        
        # For each column we need information from, get its position from the dictionary
        for name in self._column_dict.keys():
            if name not in position_dict.keys():
                raise Exception("{}:{:n} Expected to find \"{}\"".format(self._csv_file_name, self._csv_counter, name))
            self._column_dict[name].set_index(position_dict[name])
        
        if self._debug:
            print()
            print("{}:{:n}".format(self._csv_file_name, self._csv_counter))
            for name, column in self._column_dict.items():
                print("\t{} is column {:n}".format(name, column.get_index()))
        
        self._find_indexes = False

    def _append_tokens(self):
        try:
            e_level_tokens = self._e_level.get_tokens()
            jpi_tokens = self._jpi_level.get_tokens()
        except Exception as e:
            if self._debug:
                print("{}:{:n}".format(self._csv_file_name, self._csv_counter))
                print("\t{}".format(e))
            return

        self._ags_counter += 1
        self._ags_file.write("{:>6n}".format(self._ags_counter))
        self._ags_file.write("{:>11.3f}".format(float(e_level_tokens[0])))
        self._ags_file.write("{:>8.3f}".format(float(e_level_tokens[1]) * 0.001))
        self._ags_file.write("{:>8}".format(jpi_tokens[0]))
        self._ags_file.write("{:>6n}".format(0)) # placeholder
        self._ags_file.write("{:>6n}".format(1)) # placeholder
        self._ags_file.write("{:>10n}".format(0)) # placeholder
        self._ags_file.write("{:>10n}".format(0)) # placeholder
        self._ags_file.write("{:>10n}".format(0)) # placeholder
        self._ags_file.write(" &\n++")
        self._ags_file.write("{:>10.2f}".format(0)) # placeholder
        self._ags_file.write("{:>10.2f}".format(0)) # placeholder
        self._ags_file.write("{:>10.2f}".format(0)) # placeholder
        self._ags_file.write("{:>10.2f}".format(0)) # placeholder
        self._ags_file.write("{:>10.2f}".format(0)) # placeholder
        self._ags_file.write("{:>10.2f}".format(0)) # placeholder
        self._ags_file.write("\n")

    def set_debug(self, debug):
        self._debug = debug


def main():
    if len(sys.argv) != 3:
        print()
        print("\tUsage: {} <csv file to convert> <ags file to create>".format(sys.argv[0]))
        print("\tNote: any existing ags file will be overwritten")
        print()
        return

    converter = Converter(sys.argv[1], sys.argv[2])
    converter.main()

if __name__ == "__main__":
    main()
