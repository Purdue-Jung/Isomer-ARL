#! /usr/bin/python3

import sys
import os

# Python's split(",") method isn't enough
# as that method doesn't respect deliminators in quoted substrings
# NOTE: This will only treat commas NOT between a pair of quotes "" as deliminators
# NOTE: This does NOT accout for escaped quotes in the file (e.g., \"\")

#########################
#Tokenizes csv line
def tokenize(line):
    quoted = False
    token = ""
    tokens = []
    
    for char in line:
        if char == "\"":
            quoted = not quoted
            continue
        if char == "," and not quoted:
            tokens.append(token)
            token = ""
            continue
        token += char

    if token:
        tokens.append(token)
    return tokens

#########################
#Class for CSV Column Creation
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

#########################
#Energy level column 
class ELevel(CsvColumn):
    def __init__(self):
        super().__init__()
        self._energy = None
        self._uncertainty = None

    def get_tokens(self):
        if not self._value:
            raise ValueError("ELevel value none/empty")

        tokens = self._value.split()
        if len(tokens) < 2:
            raise ValueError(f"ELevel string \"{self.getvalue}\" split in fewer than 2 tokens")

        return tokens

#########################
#JPi column
class JPi(CsvColumn):
    #def __init__(self):
     #   super().__init__()

    def get_tokens(self):
        if not self._value:
            raise ValueError("JPi value none/empty")

        tokens = self._value.split()

        if not tokens:
            raise ValueError(f"JPi string \"{self.get_value}\" split into fewer than 1 token")
        jpi = tokens[0].replace("(", "").replace(")", "")
       
        #if not jpi:
            #raise ValueError("JPi string empty after pruning")

        if "," in jpi:
            raise ValueError(f"JPi string \"{self.get_value}\" contains a comma")

        if "+" not in jpi and "-" not in jpi:
            raise ValueError(f"JPi string \"{self.get_value}\" does not contain '+' or '-'")

        return [jpi]

        #if "," in tokens[0]:
         #   raise Exception("String \"{}\" contains deliminator, multiple values will be skipped".format(self._value))

        #if not ("+" in tokens[0] or "-" in tokens[0]):
         #   raise Exception("String \"{}\" does not contain \"+\" or \"-\"".format(self._value))

        #return tokens

#########################
#Converts csv to ASCII readable file (Ags)
class Converter:

    def __init__(self, csv_filename, ags_filename, debug=True):
        self.csv_filename = csv_filename
        self.ags_filename = ags_filename
        self.debug = debug

        self.csv_file = None
        self.ags_file = None

        self.csv_line_number = 0
        self.ags_entry_count = 0
        self.need_column_indexing = True
        
        self._find_indexes = True
        #self._csv_counter = 0
        #self._ags_counter = 0

        self._e_level = ELevel()
        self._jpi_level = JPi()

        self.columns = {
            "E(level)(keV)" : ELevel(),
            "Jπ(level)" : JPi(),
        }
    
    #########################
    def run(self):
        self._open_csv()
        self._open_ags()
        
        for line in self.csv_file:
            line = line.replace("?", "").replace("\xa0", "")

            # There are some characters which impede parsing
            # I'm resorting to manually pruning them
            #prunes = ("?", "\xa0")
            #for prune in prunes:
                #line = line.replace(prune, "")

            try:
                self._process_line(line.rstrip())
                #self._loop(line)
            except Exception as e:
                print(f"Error on line {self.csv_line_number}: {e}")
                break

        self._close_files()
        
    #########################
    def _open_csv(self): 
        self.csv_file = open(self.csv_filename, 'r', encoding='utf-8-sig')

    #########################
    def _open_ags(self):
        self.ags_file = open(self.ags_filename, 'w', encoding='utf-8')
        self.ags_file.writelines ( [
            "** ASCII Graphical Level Scheme file.\n",
            "** First five lines are reserved for comments.\n",
            "** This file created 03-May-00 14:47:38\n",
            "** Program GLS,  author A.Rounds \n",
            "**\n",
            " ASCII GLS file format version 1.0\n",
            "**  Z Nlevels Ngammas  Nbands Nlabels CharSizeX CharSizeY ArrowWidFact\n",
            "   00     000     000      00       0     75.00     85.00      3.00\n",
            "**  MaxArrowTan MaxArrowDX DefBandWid DefBandSep ArrowWidth ArrowLength\n",
            "       0.267900     999.00     600.00     150.00      40.00      80.00\n",
            "** ArrowBreak LevelBreak   LevCSX   LevCSY LevEnCSX LevEnCSY   GamCSX   GamCSY\n",
            "       30.00       40.00    75.00    85.00    75.00    85.00    75.00    85.00\n",
            "** Level   Energy +/- err     Jpi     K Band# LevelFlag LabelFlag EnLabFlag\n",
            "++   LabelDX   LabelDY EnLabelDX EnLabelDY  LevelDX1  LevelDX2\n"
        ])

    #########################
    def _close_files(self):
        if self.csv_file:
            self.csv_file.close()
        if self.ags_file:
            self.ags_file.close()

    #########################
    def _process_line(self, line):
        self.csv_line_number += 1

        if not line.strip():
            self.need_column_indexing = True
            return
            
        tokens = tokenize(line)

        if self.need_column_indexing:
            self._set_column_indexes(tokens)
            return

        for name, column in self.columns.items():
            
            index = column.get_index()
            
            if index >= len(tokens):
                raise IndexError(f"Expected at least {index+1} tokens, got {len(tokens)}")
            column.set_value(tokens[index])

        if self.debug:
            print(f"\n{self.csv_filename}:{self.csv_line_number}")
            for name, col in self.columns.items():
                print(f"\t{name} has token \"{col.get_value()}\"")

        self._write_ags_entry()
    
    #########################
    def _set_column_indexes(self, headers):
        header_map = {name: idx for idx, name in enumerate(headers)}
        
        for name in self.columns:
            if name not in header_map:
                raise KeyError(f"Header \"{name}\" not found in line {self.csv_line_number}")
            self.columns[name].set_index(header_map[name])

        if self.debug:
            print(f"\n{self.csv_filename}:{self.csv_line_number}")
            
            for name, col in self.columns.items():
                print(f"\t{name} is column {col.get_index()}")

        self.need_column_indexing = False

    #########################

    def _write_ags_entry(self):
        try:
            e_tokens = self.columns["E(level)(keV)"].get_tokens()
            j_tokens = self.columns["Jπ(level)"].get_tokens()
        
        except Exception as e:
            if self.debug:
                print(f"\t{e}")
            return

        self.ags_entry_count += 1

        self.ags_file.write(f"{self.ags_entry_count:>6d}")
        self.ags_file.write(f"{float(e_tokens[0]):>11.3f}")
        self.ags_file.write(f"{float(e_tokens[1]) * 0.001:>8.3f}")
        self.ags_file.write(f"{j_tokens[0]:>8}")
        self.ags_file.write(f"{0:>6d}{1:>6d}{0:>10d}{0:>10d}{0:>10d} &\n")
        self.ags_file.write("++")
        self.ags_file.write(f"{0:>10.2f}" * 6)
        self.ags_file.write("\n")

    

def main():
    if len(sys.argv) != 3:
        print(f"\nUsage: {sys.argv[0]} <csv_file> <ags_file>")
        print("Note: Existing ags file will be overwritten.\n")
        return

    converter = Converter(sys.argv[1], sys.argv[2])
    converter.run()

if __name__ == "__main__":
    main()
