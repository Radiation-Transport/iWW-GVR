"""
########################################################################################################
# Copyright 2019 F4E | European Joint Undertaking for ITER and the Development                         #
# of Fusion Energy (‘Fusion for Energy’). Licensed under the EUPL, Version 1.2                         #
# or - as soon they will be approved by the European Commission - subsequent versions                  #
# of the EUPL (the “Licence”). You may not use this work except in compliance                          #
# with the Licence. You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl.html       #
# Unless required by applicable law or agreed to in writing, software distributed                      #
# under the Licence is distributed on an “AS IS” basis, WITHOUT WARRANTIES                             #
# OR CONDITIONS OF ANY KIND, either express or implied. See the Licence permissions                    #
# and limitations under the Licence.                                                                   #
########################################################################################################
Author: Alvaro Cubi
"""

import os
from copy import deepcopy
from typing import List

from iww_gvr.weight_window import WW

MAIN_MENU = """
 ***********************************************
        Weight window manipulator and GVR
 ***********************************************

 * Open weight window file   (open)   
 * Display ww information    (info)   
 * Write ww                  (write)
 * Export as VTK             (vtk)
 * Analyse                   (analyse)
 * Plot                      (plot)    
 * Weight window operation   (operate)
 * GVR generation            (gvr)
 * Exit                      (end)    
"""

OPERATE_MENU = """             
 * Softening and normalize   (soft)
 * Add                       (add)
 * Remove                    (rem)
 * Mitigate long histories   (mit)
 * Exit                      (end)
"""


def clear_screen():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')


class Menu:
    def __init__(self):
        self.ww_list: List[WW] = []
        self.go_main_menu()

    @property
    def ww_filenames(self):
        return [ww.filename for ww in self.ww_list]

    def go_main_menu(self, subtext=None):
        clear_screen()
        print(MAIN_MENU)
        if subtext:
            print(subtext)
        command = input(" enter action: ")
        self.process_main_command(command)

    def process_main_command(self, command):
        if command == 'open':
            return self.command_open()
        elif command == 'info':
            return self.command_info()
        elif command == 'write':
            return self.command_write()
        elif command == 'vtk':
            return self.command_vtk()
        elif command == 'analyse':
            return self.command_analyse()
        elif command == 'plot':
            return self.command_plot()
        elif command == 'operate':
            return self.go_operate_menu()
        elif command == 'gvr':
            return self.command_gvr()
        elif command == 'end':
            print('Thank you for using iww_gvr, see you soon!')
        else:
            return self.go_main_menu("Not a valid action...")

    def go_operate_menu(self, subtext=None):
        clear_screen()
        print(OPERATE_MENU)
        if subtext:
            print(subtext)
        command = input(" enter action: ")
        self.process_operate_command(command)

    def process_operate_command(self, command):
        if command == 'soft':
            return self.command_soft()
        elif command == 'add':
            return self.command_add()
        elif command == 'rem':
            return self.command_rem()
        elif command == 'mit':
            return self.command_mit()
        elif command == 'end':
            return self.go_main_menu()
        else:
            return self.go_operate_menu("Not a valid action...")

    def select_ww_index(self):
        # In case there is only 1 ww loaded there is no need to ask
        if len(self.ww_list) == 1:
            return 0

        for i in range(len(self.ww_list)):
            print(f"[{i}]: {self.ww_list[i].filename}")
        inp = input(" select the ww by index: ")
        try:
            idx = int(inp)
            if idx not in [x for x in range(len(self.ww_list))]:
                print("Index not among possible options...")
                return self.select_ww_index()
            return idx
        except ValueError:
            print("Not a valid index...")
            return self.select_ww_index()

    # Commands Main Menu ##############################################

    def command_open(self):
        filename = input(" enter ww file name: ")
        if filename in self.ww_filenames:
            return self.go_main_menu('WW filename already loaded...')
        try:
            ww = WW.read_from_ww_file(filename)
            self.ww_list.append(ww)
            return self.go_main_menu('WW file loaded!')
        except FileNotFoundError:
            return self.go_main_menu("File not found for this path or filename...")

    def command_info(self):
        if len(self.ww_list) == 0:
            return self.go_main_menu("No WWs loaded...")
        idx = self.select_ww_index()
        ww = self.ww_list[idx]
        return self.go_main_menu(ww.info)

    def command_write(self):
        if len(self.ww_list) == 0:
            return self.go_main_menu("No WWs loaded...")
        idx = self.select_ww_index()
        ww = self.ww_list[idx]
        ww.write_ww_file(ww.filename + '_written')
        return self.go_main_menu('WW file written!')

    def command_vtk(self):
        if len(self.ww_list) == 0:
            return self.go_main_menu("No WWs loaded...")
        idx = self.select_ww_index()
        ww = self.ww_list[idx]
        ww.export_to_vtk(ww.filename + '_VTK')
        return self.go_main_menu('VTK file written!')

    def command_analyse(self):
        if len(self.ww_list) == 0:
            return self.go_main_menu("No WWs loaded...")
        idx = self.select_ww_index()
        ww = self.ww_list[idx]
        return self.go_main_menu(ww.info_analyse)

    def command_plot(self):
        if len(self.ww_list) == 0:
            return self.go_main_menu("No WWs loaded...")
        idx = self.select_ww_index()
        ww = self.ww_list[idx]
        ww.plotter.plot_interactive_plane(ww.particles[0], ww.energies[ww.particles[0]][0])
        return self.go_main_menu()

    def command_gvr(self):
        filename = input(" enter meshtally file name: ")
        if filename in self.ww_filenames:
            return self.go_main_menu('WW filename already loaded...')
        try:
            tally_id = int(input(" enter the tally id: "))
            ww = WW.read_from_meshtally_file(filename, tally_id)
            self.ww_list.append(ww)
            return self.go_main_menu('GVR produced!')
        except FileNotFoundError:
            return self.go_main_menu("File not found for this path or filename...")
        except (ValueError, KeyError):
            return self.go_main_menu("File or tally id were incorrect...")

    # Commands Operate Menu ##############################################

    def command_soft(self):
        if len(self.ww_list) == 0:
            return self.go_operate_menu("No WWs loaded...")
        idx = self.select_ww_index()
        ww = deepcopy(self.ww_list[idx])
        ww.filename = input('Write name of the new WW: ')
        try:
            norm = float(input('Enter normalization factor: '))
            soft = float(input('Enter softening factor: '))
        except ValueError:
            return self.go_main_menu('The factor introduced was not a valid number...')
        ww.apply_normalization(norm)
        ww.apply_softening(soft)
        return self.go_main_menu(f'Normalization and softening factors applied to {ww.filename}!')

    def command_add(self):
        if len(self.ww_list) == 0:
            return self.go_operate_menu("No WWs loaded...")
        idx = self.select_ww_index()
        ww = self.ww_list[idx]
        if len(ww.particles) == 2:
            return self.go_operate_menu('The WW already had 2 particle types...')
        try:
            norm = float(input('Enter normalization factor: '))
            soft = float(input('Enter softening factor: '))
        except ValueError:
            return self.go_main_menu('The factor introduced was not a valid number...')
        ww.add_particle(norm=norm, soft=soft)
        return self.go_main_menu(ww.info)

    def command_rem(self):
        if len(self.ww_list) == 0:
            return self.go_operate_menu("No WWs loaded...")
        idx = self.select_ww_index()
        ww = self.ww_list[idx]
        if len(ww.particles) == 1:
            return self.go_operate_menu('The WW already has only 1 particle type...')
        ww.remove_particle()
        return self.go_main_menu(ww.info)

    def command_mit(self):
        if len(self.ww_list) == 0:
            return self.go_operate_menu("No WWs loaded...")
        idx = self.select_ww_index()
        ww = self.ww_list[idx]
        try:
            max_ratio = float(input(" enter the maximum ratio allowed: "))
            ww.filter_ratios(max_ratio)
            return self.go_main_menu(ww.info_analyse+' Mitigation completed!')
        except ValueError:
            return self.go_operate_menu("Maximum ratio was invalid...")


if __name__ == '__main__':
    Menu()
