#! /usr/bin/python3

import sys
import os
import re
import string
import datetime

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
  
    def get_tokens(self):
        if not self._value:
            raise ValueError("ELevel value none/empty")
        tokens = self._value.split()
        
        if len(tokens) == 1:
            tokens.append("0")  #Default uncertainty
        elif len(tokens) < 2:
            raise ValueError(f"ELevel string \"{self.get_value()}\" split in <2 tokens")

        return tokens

#########################
#JPi column
class JPi(CsvColumn):
 
    def get_tokens(self):
        if not self._value:
            raise ValueError("JPi value none/empty")

        tokens = self._value.split()
        if not tokens:
            raise ValueError(f"JPi string \"{self.get_value()}\" split into <1 token")

        ##If you want more data USE THIS SECTION. Utilizes () qualifiers. 
        '''
        jpi = tokens[0].replace("(", "").replace(")", "")
        if "," in jpi:
            raise ValueError(f"JPi string \"{self.get_value()}\" contains a comma")

        if "+" not in jpi and "-" not in jpi:
            raise ValueError(f"JPi string \"{self.get_value()}\" does not contain '+' or '-'")
        '''
        ##This section only utilizes data formatted as number+ or number-. 
        jpi = tokens[0].strip()
        #Strict pattern check: must be digit(s) followed by '+' or '-'
        if not re.fullmatch(r"(?:\d+|\d+/\d+)[+-]", jpi):
            raise ValueError(f"JPi string \"{self.get_value()}\" not proper format")

        return [jpi]

#########################
#Gamma energy column
class EGamma(CsvColumn):
    
    def get_tokens(self):
        if not self._value:
            raise ValueError("EGamma value none/empty")
        tokens = self._value.split()
        
        if len(tokens) == 1:
            tokens.append("0")  # Default uncertainty
        elif len(tokens) < 2:
            raise ValueError(f"EGamma string \"{self.get_value()}\" split in <2 tokens")

        return tokens

#########################
#Intensity column
class Intensity(CsvColumn):
    def get_tokens(self):
        tokens = self._value.split()
        
        if len(tokens) == 1:
            tokens.append("0")
        elif len(tokens) < 2:
            raise ValueError(f"Intensity string \"{self.get_value()}\" split into <2 tokens")
        
        return tokens

#########################
#Multipolarity column
class Multipolarity(CsvColumn):
    def get_tokens(self):
        
        token = self._value.strip()
        if not token:
            raise ValueError("Multipolarity value is empty")
        
        return [token]

#########################
#Converts csv to ASCII readable file (Ags)
class Converter:
    #Intialization for entire file
    def __init__(self, csv_filename, ags_filename, debug=True, delimiter=','):
        self.delimiter = delimiter
        self.csv_filename = csv_filename
        self.ags_filename = ags_filename
        self.debug = debug
        #self.ags_file = open(self.ags_filename, 'w')
        
        self.csv_file = None
        self.ags_file = None

        self.csv_line_number = 0
        self.ags_entry_count = 0
        self.need_column_indexing = True
        
        #Level Columns
        self.columns = {
            "E(level)(keV)" : ELevel(),
            "JPi(level)" : JPi(),
        }
        
        #Gamma Columns
        self.gamma_columns = {
            "E()(keV)": EGamma(),
            "I()": Intensity(),
            "M()": Multipolarity(),
        }

        self.band_info_columns = {
            "BandName": CsvColumn(),
            "BandNum": CsvColumn(),
            "Z": CsvColumn(),
        }
        
        self.ags_entries = []
        self.band_entries = []
        self.gamma_entries = []
        
    #########################
    def run(self):
        self.csv_file = open(self.csv_filename, 'r', encoding='utf-8-sig')
        self.ags_file = open(self.ags_filename, 'w', encoding='utf-8')
        
        for line in self.csv_file:
            line = line.replace("?", "").replace("\xa0", "")
            try:
                self._process_line(line.rstrip())
            except Exception as e:
                print(f"Error on line {self.csv_line_number}: {e}")
                break
        
        self._write_header()
        self._write_ags_section()
        self._write_band_section()
        self._write_gamma_section()
        
        #if self.csv_file:
        self.csv_file.close()
        #if self.ags_file:
        self.ags_file.close()

    #########################
    def _write_header(self):

        #Extract bands & z from BandNum in csv file
        nbands = 0  
        z_val = 0
        
        try:
            #Had to reopen csv file like this due to the damn Z value not wanting to format
            with open(self.csv_filename, 'r', encoding='utf-8-sig') as f:
                header_line = f.readline().strip() 
                header = header_line.split(self.delimiter) 
                headers = {name.strip(): idx for idx, name in enumerate(header)} 

                #Value definitions
                bandnum_idx = headers.get("BandNum")
                z_idx = headers.get("Z")

                for line in f:
                    tokens = line.strip().split(self.delimiter)
                    if len(tokens) <= max(bandnum_idx, z_idx):
                        continue

                    band_val = tokens[bandnum_idx].strip()
                    z_val_str = tokens[z_idx].strip()

                    #Confirm the values are formatted properly
                    if band_val:
                        nbands = int(float(band_val))       # 18.0 -> 18
                    if z_val_str:
                        z_val = int(float(z_val_str))  # 72.0 -> 72
                    break  # use first valid line only
                
        except Exception as e:
            if self.debug:
                print(f"[HEADER] Failed to extract BandNum or Z: {e}")
        
        self.ags_file.writelines([
            "** ASCII Graphical Level Scheme file.\n",
            "** First five lines are reserved for comments.\n",
            f"** This file created {datetime.datetime.now().strftime("%d-%b-%y %H:%M:%S")}\n",
            "** Program GLS,  author A.Rounds \n", 
            "**\n",
            " ASCII GLS file format version 1.0\n",
            "**  Z Nlevels Ngammas  Nbands Nlabels CharSizeX CharSizeY ArrowWidFact\n",
            f"{z_val:5d}"                     #Z-value
            f"{len(self.ags_entries):8d}"     #Nlevels
            f"{len(self.gamma_entries):8d}"   #Ngammas
            f"{nbands:8d}"                    #Nbands
            f"{0:8d}"              
            f"{75.00:10.2f}"
            f"{85.00:10.2f}"
            f"{3.00:10.2f}\n"
        ])

        if self.debug:
            print(
                f"[HEADER] Final values - Z: {z_val}, Nlevels: {len(self.ags_entries)}, "
                  f"Ngammas: {len(self.gamma_entries)}, Nbands: {nbands}"
            )


    #########################
    def _write_ags_section(self):

        self.ags_file.writelines([
            "**  MaxArrowTan MaxArrowDX DefBandWid DefBandSep ArrowWidth ArrowLength\n",
            f"{0.267900:15.6f}"
            f"{999.00:11.2f}"
            f"{600.00:11.2f}"
            f"{150.00:11.2f}"
            f"{40.00:11.2f}"
            f"{80.00:11.2f}\n",
            "** ArrowBreak LevelBreak   LevCSX   LevCSY LevEnCSX LevEnCSY   GamCSX   GamCSY\n",
            f"{30.00:12.2f}" #Arrow Break
            f"{40.00:12.2f}"
            f"{75.00:9.2f}" #LevCSX
            f"{85.00:9.2f}"
            f"{75.00:10.2f}"
            f"{85.00:10.2f}"
            f"{75.00:10.2f}"
            f"{85.00:10.2f}\n",
            "** Level   Energy +/- err     Jpi     K Band# LevelFlag LabelFlag EnLabFlag\n",
            "++   LabelDX   LabelDY EnLabelDX EnLabelDY  LevelDX1  LevelDX2\n"
        ])

        
        for entry in self.ags_entries:
            self.ags_file.write(
                f"{entry['index']:>6d}"
                f"{entry['energy']:>11.3f}"
                f"{entry['energy_err']:>8.3f}"
                f"{entry['jpi']:>8}"
                f"{0:>6d}{1:>6d}{0:>10d}{0:>10d}{0:>10d} &\n"
            )
            self.ags_file.write(
                f"++{0.00:10.2f}{0.00:10.2f}{0.00:10.2f}{0.00:10.2f}"
                f"{0.00:10.2f}{0.00:10.2f}\n"
            )
        '''
        try:
            e_tokens = self.columns["E(level)(keV)"].get_tokens()
            j_tokens = self.columns["JPi(level)"].get_tokens()
        
        except Exception as e:
            if self.debug:
                print(f"\t{e}")
            return

        self.ags_entry_count += 1
        '''
        #self.ags_file.write(f"{self.ags_entry_count:>6d}")
        #self.ags_file.write(f"{float(e_tokens[0]):>11.3f}")
        #self.ags_file.write(f"{float(e_tokens[1]) * 0.001:>8.3f}")
        #self.ags_file.write(f"{j_tokens[0]:>8}")
        #self.ags_file.write(f"{0:>6d}{1:>6d}{0:>10d}{0:>10d}{0:>10d} &\n")
        #self.ags_file.write("++")
        #self.ags_file.write(f"{0:>10.2f}" * 6)
        #self.ags_file.write("\n")
    
    #########################
    def _write_band_section(self):
        
        if self.debug:
            print("Writing Band section...")

        try:
            #Reset band entries each time this runs
            self.band_entries = []
            #Read BandName or BandNum from first row
            self.csv_file.seek(0)
            header = next(self.csv_file).strip().split(self.delimiter)
            if self.debug:
                print(f"[DEBUG] Parsed header: {header}")
            headers = {name.strip(): idx for idx, name in enumerate(header)}

            bandname_col = headers.get("BandName")
            bandnum_col = headers.get("BandNum")
            #z_col = headers.get("Z")

            if bandname_col is None or bandnum_col is None:
                raise KeyError("BandName or BandNum missing")

            # Find first valid BandName/BandNum entry
            for line in self.csv_file:
                tokens = line.strip().split(self.delimiter)
                if len(tokens) <= max(bandname_col, bandnum_col):
                    continue

                band_name = tokens[bandname_col].strip()
                band_num_str = tokens[bandnum_col].strip()
                #z_num_str = tokens[z_col].strip()

                if not band_name or not band_num_str:
                    continue

                band_count = int(float(band_num_str))  # 18.0 -> 18
                #z_count = int(float(z_num_str))        #72.0 -> 72

                #Generate suffixes a, b, ect for band_name
                suffixes = [string.ascii_lowercase[i] if i < 26 else string.ascii_lowercase[(i - 26) // 26] + string.ascii_lowercase[(i - 26) % 26] for i in range(band_count)]

                for s in suffixes:
                    self.band_entries.append({
                        'name': f"{band_name}{s}",
                        'x0': 00000.00,
                        'nx': 600.00,
                        'label_dx': 0.00,
                        'label_dy': 0.00,
                        'enlabel_dx': 0.00,
                        'enlabel_dy': 0.00,
                    })
                break  # done after first valid entry

            if not self.band_entries:
                print("No band entry.")
                return

            #Write header line
            self.ags_file.writelines([
                "** Band   Name        X0        NX   LabelDX   LabelDY EnLabelDX EnLabelDY\n"
            ])

            #Write each band line
            for idx, entry in enumerate(self.band_entries, start=1):
                self.ags_file.write(
                    f"{idx:6d}"
                    f"{entry['name']:>10s}"
                    f"{entry['x0']:10.2f}"
                    f"{entry['nx']:10.2f}"
                    f"{entry['label_dx']:10.2f}"
                    f"{entry['label_dy']:10.2f}"
                    f"{entry['enlabel_dx']:11.2f}"
                    f"{entry['enlabel_dy']:11.2f}\n"
                )

        except Exception as e:
            print(f"Error while writing Band section: {e}")

            
    #########################
    def _write_gamma_section(self):
        self.ags_file.writelines([
            "** Gamma   Energy +/- err  Mult  ILev  FLev  Intensity +/- err\n",
            "++     ConvCoef +/- error      BrRatio +/- error     MixRatio +/- error\n",   
            "++   GammaX1  GammaX2  LabelDX  LabelDY GammaFlag LabelFlag\n"
        ])

        sorted_gammas = sorted(self.gamma_entries, key=lambda g: g["energy"])

        for idx, g in enumerate(sorted_gammas, start=1):
            #line 1: gamma data
            self.ags_file.write(
                f"{idx:>6d}"
                f"{g['energy']:>10.3f}"
                f"{g['energy_err']:>9.3f}"
                f"{g['mult']:>5}"
                f"{g['ilev']:>6d}{g['flev']:>6d}"
                f"{g['intensity']:>10.4f}"
                f"{g['intensity_err']:>9.4f} &\n"
            )
            #line 2: conv, br, mix ratios
            self.ags_file.write(
                f"++  {6.46410E-01:>11.5E} {0.000E+00:>10} "
                f"{0.00000E+00:>13} {0.000E+00:>10} "
                f"{0.00000E+00:>13} {0.000E+00:>10} &\n"
            )
            #line 3: gamma flags, labels
            self.ags_file.write(
                f"++  {0.00:>10.2f} {62282.00:>10.2f}" 
                f"{0.00:>10.2f} {0.00:>10.2f}"
                f"{0:>10d}{0:>10d}\n"
            )

    #########################
    #Organizes Processes
    def _process_line(self, line):
        self.csv_line_number += 1
        tokens = line.strip().split(self.delimiter)

        if not any(tokens):
            self.need_column_indexing = True
            return

        if self.need_column_indexing:
            self._set_column_indexes(tokens)
            return
            
        #Level columns
        for name, column in self.columns.items():
            index = column.get_index()
            if index >= len(tokens):
                raise IndexError(f"Expected at least {index+1} tokens, got {len(tokens)}")
            column.set_value(tokens[index])
        
        #Gamma columns
        for name, column in self.gamma_columns.items():
            index = column.get_index()
            if index >= len(tokens):
                raise IndexError(f"Expected at least {index+1} tokens, got {len(tokens)}")
            column.set_value(tokens[index])
        
        #Band columns
        for name, column in self.band_info_columns.items():
            index = column.get_index()
            if index >= len(tokens):
                raise IndexError(f"Expected at least {index+1} tokens, got {len(tokens)}")
            column.set_value(tokens[index])

        #Self debug 
        if self.debug:
            print(f"\n{self.csv_filename}:{self.csv_line_number}")
            for name, col in {**self.columns, **self.gamma_columns}.items():
                print(f"\t{name} has token \"{col.get_value()}\"")
        
           
        #self._write_ags_section()
        if self._collect_level_entry():
            self._write_gam_entry()
        
        
    #########################
    #Defines column indexes
    #Alter this to accept new headers
    def _set_column_indexes(self, headers):
        header_map = {name.strip(): idx for idx, name in enumerate(headers)}

        #For ELevel & JPi
        for name in self.columns:
            if name not in header_map:
                raise KeyError(f"Header \"{name}\" not found in line {self.csv_line_number}")
            self.columns[name].set_index(header_map[name])

        #For EGamma, Intensity, Multipolarity
        for name in self.gamma_columns:
            if name not in header_map:
                raise KeyError(f"Header \"{name}\" not found in line {self.csv_line_number}")
            self.gamma_columns[name].set_index(header_map[name])
            
        #For BandName & BandNum
        for name in self.band_info_columns:
            if name not in header_map:
                raise KeyError(f"Header \"{name}\" not found in line {self.csv_line_number}")
            self.band_info_columns[name].set_index(header_map[name])
        
        
        self.need_column_indexing = False
        
        '''
        if self.debug:
            print(f"\n{self.csv_filename}:{self.csv_line_number}")
            for name, col in {**self.columns, **self.gamma_columns}.items():
                print(f"\t{name} is column {col.get_index()}")
        '''
         
    #########################
    def _write_gam_entry(self):
        # Create gamma entry based on level transition
        try:
            g_tokens = self.gamma_columns["E()(keV)"].get_tokens()
            i_tokens = self.gamma_columns["I()"].get_tokens()
            m_tokens = self.gamma_columns["M()"].get_tokens()
            
            if len(self.ags_entries) < 2: #not enought to define transition
                return
                
            self.gamma_entries.append ({
                "energy": float(g_tokens[0]),
                "energy_err": float(g_tokens[1]) * 0.001,
                "mult": m_tokens[0],
                "ilev": self.ags_entries[-1]["index"],
                "flev": self.ags_entries[-2]["index"],
                "intensity": float(i_tokens[0]),
                "intensity_err": float(i_tokens[1]),
            })
            
            #self.gamma_entries.append(gamma_entry)
    
        except Exception as e:
            if self.debug:
                print(f"\t{e}")
    
    #########################
    def _collect_level_entry(self):
        try:
            e_tokens = self.columns["E(level)(keV)"].get_tokens()
            j_tokens = self.columns["JPi(level)"].get_tokens()

            self.ags_entries.append({
                "index": self.ags_entry_count + 1,
                "energy": float(e_tokens[0]),
                "energy_err": float(e_tokens[1]) * 0.001,
                "jpi": j_tokens[0],
            })
            
            #self.ags_entries.append(level_entry)
            self.ags_entry_count += 1
            return True ##

        except Exception as e:
            if self.debug:
                print(f"\t[Invalid Level] {e}")
            return False

#########################
def main():

    if len(sys.argv) < 3:
        print("Usage: convert-Copy.py <csv_file> <ags_file>")
        #print("Usage: script.py input.csv output.ags")
        sys.exit(1)
        
    Converter(sys.argv[1], sys.argv[2], debug=True, delimiter=',').run()


    #csv_filename = sys.argv[1]
    #ags_filename = sys.argv[2]

    #converter = Converter(csv_filename, ags_filename, debug=True)
    #converter.run()
    
if __name__ == "__main__":
    main()
