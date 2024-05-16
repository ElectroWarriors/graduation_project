import os
import math
import time
import random
import numpy as np
import pandas as pd

from PyLTSpice import RawRead
from PyLTSpice import SimRunner
from PyLTSpice import SpiceEditor


class MOS_Operating_Point:
    def __init__(self):
        self.data = {"others": {} } # others 处存放其他仿真信息（如{others: {"gain": xxx}, {"BW": xxx}, ... }）

    def add_element(self, mosx, argx, value):
        if(mosx not in self.data):
            self.data[mosx] = {}
        self.data[mosx][argx] = value

    def read_element(self, mosx, argx):
        if( mosx in self.data and argx in self.data[mosx] ):
            return self.data[mosx][argx]
        else:
            return None
        


class Simulator:
    def __init__(self, SimFilePath, SimFileName, LTspiceExec, Model="smic18"): # W_designed, L_designed, R_designed, C_designed, 
        self.num_sim = 0
        self.SimFilePath = SimFilePath
        self.SimFileName = SimFileName
        self.LTC = SimRunner(simulator=LTspiceExec, output_folder=SimFilePath+'temp')
        self.LTC.create_netlist(SimFilePath+SimFileName+".asc")
        self.netlist = SpiceEditor(SimFilePath+SimFileName+".net")
        self.model = self.ModelSelection(Model)
        print(self.model["inc_lib"])
        self.netlist.add_instruction(".include " + SimFilePath + "model\\"+ self.model["inc_lib"])
        self.netlist.set_component_value("V3", self.model["VDD"])

        for mosi in self.netlist.get_components("M"):
            mosi_info = self.netlist.get_component_info(mosi)
            if(mosi_info["value"] == "lib_nmos"):
                self.netlist.set_element_model(mosi, self.model["n_model"])
            elif(mosi_info["value"] == "lib_pmos"):
                self.netlist.set_element_model(mosi, self.model["p_model"])
            else:
                print(mosi, ":", mosi_info, "error")
                raise ValueError("mos lib name error")
        self.netlist.save_netlist(SimFilePath+SimFileName+".net")

        # self.W_designed = W_designed
        # self.L_designed = L_designed
        # self.R_designed = R_designed
        # self.C_designed = C_designed
    
    def ModelSelection(self, Model):
        if(isinstance(Model, dict)):
            if(("inc_lib" in Model.keys()) and ("n_model" in Model.keys()) and 
               ("p_model" in Model.keys()) and ("VDD"     in Model.keys())):
                return Model
            else:
                raise ValueError("Dict do not contain enough items.")
        
        if(Model == "smic18"):
            inc_lib, n_model, p_model, VDD = "smic18_mos_model.lib", "smic18n33", "smic18p33", 3.3
        elif(Model == "tsmc18"):
            inc_lib, n_model, p_model, VDD = "tsmc18_mos_model.lib", "tsmc_nch_rf33", "tsmc_pch_rf33", 3.3
        elif(Model == "smic13"):
            inc_lib, n_model, p_model, VDD = "smic13_mos_model.lib", "smic13n33", "smic13p33", 3.3
        # smic35
        elif(Model == "smic35"):
            inc_lib, n_model, p_model, VDD = "smic35_mos_model.lib", "smic35n33", "smic35p33", 3.3
        # umc55n
        elif(Model == "umc55n"):
            inc_lib, n_model, p_model, VDD = "l55lp_v0112.lib", "n_12_lplvt", "p_12_lplvt", 3.3
        else:
            print(Model)
            raise ValueError("No such model.")
        Model_dict = {"inc_lib": inc_lib, "n_model": n_model, "p_model": p_model, "VDD": 3.3}
        return Model_dict
    
    def MOS_OP_Read(self, filepath):
        """ 
        # read MOS oierating point from .log file
        # input: file address with file name
        # eg: filepath = "C:\\Users\\HJH\\Desktop\\simulator\\based_on_python\\OPAMP1\\simfile\\temp\\opamp_1.log"

        # output: MOS_operating_point - a dictionary with the element of dictionary { "xxx": {}, "xxx": {}, ... }
        """
        MOS_operating_point = MOS_Operating_Point()
        flag = 0
        MOS_list = []
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
            for line in file:
                # print(line.strip())
                if(line.find("Semiconductor Device Operating Points") != -1):
                    flag = 1
                if(flag < 1):
                    continue
                if(line.startswith("Name:")):
                    MOS_list = line.split()
                    flag = 2
                    # print("tab     ",end="")
                    # print(MOS_list)
                elif(line.startswith(" ")):
                    continue
                else:
                    if(flag >= 2):
                        value = line.split()
                        # print(line)
                        if(len(MOS_list) == len(value)):
                            value[0] = value[0].replace(":","")
                            # print(value)
                            for i in range(1,len(MOS_list)):
                                MOS_operating_point.add_element(MOS_list[i], value[0], value[i])
                            if(line.startswith("Cbs:")):
                                flag = 1
                        # else:
                            # print(line)
                    else: # others
                        if(line.startswith("bw=")):
                            BW = self.BW_Read(line)
                            MOS_operating_point.add_element("others", "BW", BW)
                        elif(line.startswith("gbw=")):
                            GBW = self.GBW_Read(line)
                            MOS_operating_point.add_element("others", "GBW", GBW)
                            # print("GBW: ", GBW)
                        elif(line.startswith("delta_phase:")):
                            AvodB_PMdeg = self.AvodB_PMdeg_Read(line)
                            # print("Avo and PM: ", AvodB_PMdeg)
                            MOS_operating_point.add_element("others", "AvodB", AvodB_PMdeg[0])
                            MOS_operating_point.add_element("others", "PMdeg", AvodB_PMdeg[1])
        MOS_operating_point = self.MOS_OperatingRegion_Power(MOS_operating_point)
        return MOS_operating_point

    def BW_Read(self, line):
        """
        # read BW from a line in .log file
        # example: bw=45595.6 FROM 10 TO 45605.6
        """
        BW = line.split()[0].split("=")[1]
        return BW

    def GBW_Read(self, line):
        """
        # read BW from a line in .log file
        # example: bw=45595.6 FROM 10 TO 45605.6
        """
        GBW = line.split()[0].split("=")[1]
        return GBW

    def AvodB_PMdeg_Read(self, line):
        """
        # read Avo and PM from a line in .log file
        # example: delta_phase: vo0_a/v(vout)=(89.6239dB,2.09594? at 8.92756e+07
        """
        AvodB_PMdeg = line.split("=(")[1].split(")")[0].split("dB,")
        return AvodB_PMdeg


    def MOS_OperatingRegion_Power(self, MOS_operating_point):
        """
        # add MOS operating region to the MOS_operating_point
        """
        satuation = []
        others = []
        P0 = 0
        for mosx in MOS_operating_point.data:
            if(mosx.startswith("m") == False):
                continue
            model = MOS_operating_point.read_element(mosx, "Model")
            Vth = float(MOS_operating_point.read_element(mosx, "Vth"))
            Vgs = float(MOS_operating_point.read_element(mosx, "Vgs"))
            Vds = float(MOS_operating_point.read_element(mosx, "Vds"))
            Vsat = float(MOS_operating_point.read_element(mosx, "Vdsat"))
            Id   = float(MOS_operating_point.read_element(mosx, "Id"))

            P0 = P0 + abs(Vds*Id)
            
            if(model == self.model["n_model"]):
                if( (Vgs > Vth) and (Vds >= Vgs-Vth) ):
                    satuation.append(mosx)
                else:
                    others.append(mosx)
                    # print(Vgs)
                    # print(Vth)
                    # print(Vds)
            else:
                if( (Vgs < Vth) and (Vds <= Vgs-Vth) ):
                    satuation.append(mosx)
                else:
                    others.append(mosx)
            # region = {"satuation": satuation, "others": others}
        MOS_operating_point.add_element("region", "satuation", satuation)
        MOS_operating_point.add_element("region", "others", others)
        MOS_operating_point.add_element("others", "P0", P0)
        return MOS_operating_point

    # Design of the Channel Width and Channel Length of MOS Transistors
    def a_WL_setting(self, netlist, a_Length_nm, a_Width_nm):
        """
        eg:
            a_Length_nm = [1, 3, 5] (Unit: nm)
            a_Width_nm  = [2, 4, 1] (Unit: nm)
        """
        num_MOS = len(netlist.get_components("M"))
        for i in range(0, num_MOS):
            netlist.set_parameter("l_m" + str(i), str(a_Length_nm[i]) + "n")
            netlist.set_parameter("w_m" + str(i), str(a_Width_nm[i])  + "n")
        return netlist


    def Simulator(self, a_Width_nm, a_Length_nm, R, Cc):
        start = time.time()
        SimFilePath = self.SimFilePath
        SimFileName = self.SimFileName

        print("\033[1;36m", "a_Length_nm:", "\033[0m", a_Length_nm)
        print("\033[1;36m", "a_Width_nm:",  "\033[0m", a_Width_nm)
        print("\033[1;36m", "R:",           "\033[0m", R)
        print("\033[1;36m", "Cc:",          "\033[0m", Cc)
        # print("time:", time.time()-start)
        netlist = self.netlist
        # print("time:", time.time()-start)
        sim_result = {}
        # analysis_type = 0: only We can obtain the DC operating point information of the MOS transistor from the .log file
        # analysis_type = 1: .DC
        # analysis_type = 2: .TRAN, We cannot obtain the DC operating point information of the MOS transistor from the .log file
        # analysis_type = 3:    We can obtain the DC operating point information of the MOS transistor from the .log file
        #                       We can also obtain the AC characteristics from the .raw file
        for analysis_type in [3]:
            # print("analysis:", time.time()-start)
            # set model
            # netlist.set_element_model("M1", "smic18n33")
            # set W/L (array from M0 to M12) （unit： nm）, set R
            netlist = self.a_WL_setting(netlist, a_Length_nm, a_Width_nm)
            netlist.set_component_value("R0", str(R))
            netlist.set_component_value("Cc", str(Cc))
            # set command
            # .OP 
            netlist.add_instruction(".op")
            
            # netlist.add_instruction(".measure AVG P(VDD) FROM=0 TO=1m") # failed
            # 其实测功耗只要读I(V3)即可 P = VDD*I(V3)
            if (analysis_type == 0):
                netlist.add_instruction(".save V(Vinp) V(Vinn) V(Vout) I(V3)")
            if (analysis_type == 1): # DC
                print()
            elif(analysis_type == 2): # TRAN
                netlist.set_element_model('V1', "SINE(2.2 10u 100k)")
                netlist.set_element_model('V2', "SINE(2.2 -10u 100k)")
                netlist.add_instruction(".tran 0 100u 0 1u")
            elif(analysis_type == 3): # AC
                netlist.add_instruction(".ac oct 100 1 1G")
                # 计算3dB带宽
                netlist.add_instruction(".meas AC Vo0_A FIND mag(V(Vout)) AT 1")
                netlist.add_instruction(".meas AC BW TRIG mag(V(Vout))=Vo0_A TARG mag(V(Vout))=Vo0_A/sqrt(2) FALL=1")
                # 计算增益带宽积
                netlist.add_instruction(".meas GBW TARG mag(V(Vout)/V(Vinp)/2)=1 FALL=1")
                # 计算相位裕度 angle>0，相位裕度为正（一般要求＞45°）
                netlist.add_instruction(".meas AC Vo0 FIND V(Vout) AT 1")
                netlist.add_instruction(".meas AC delta_phase FIND Vo0_A/V(Vout) WHEN mag(V(Vout)/V(Vinp)/2)=1")
                netlist.add_instruction(".meas AC V(Vout)/V(Vinp)/2")
                # netlist.add_instruction(".save frequency I(V3) V(Vout) V(Vinp) V(Vout)/V(Vinp)/2")
            # start simulation
            self.LTC.run(netlist)
            self.LTC.wait_completion() #等待模拟结束
            # --------------------------READ------------------------------ #
            # print("analysistype: " + str(analysis_type)) 
            for raw, log in self.LTC:
                # print("Raw file: %s, Log file: %s" % (raw, log))
                if(raw == ""):
                    print(".raw and .log file do not exist.")
                    return False
                # print("Raw file: ", raw)
                # print("Log file: ", log)
            # 读取电路静态工作点(.raw)
            # rawdata = RawRead(raw) 
            # raldata_excelpath = SimFilePath+"temp\\" + SimFileName+"_rawdata.xlsx"
            # rawdata.to_excel(raldata_excelpath)
            # rawdata_frame = rawdata.to_dataframe()

            if(analysis_type == 0):
                print("op")
                # MOS_operating_point = process.MOS_OP_Read(log) # 读取MOS管的静态工作点(.log)
                # P0 = abs(3.3 * rawdata_frame.loc[0, "I(V3)"]) # P = rawdata_frame.loc[0].loc["I(V3)"]
                # print("P0: ", P0*1000, "mW")
                # sim_result["P0"] = P0
            elif(analysis_type == 1):
                print("dc")
            elif(analysis_type == 2):
                print("tran")
                if(False):
                    Time = rawdata.get_wave("time")
                    Vout = rawdata.get_wave("V(vout)")
                    plt.plot(Time,Vout)
                    plt.show()
            elif(analysis_type == 3):
                print("ac")
                if(False):
                    Vout_freq = rawdata_frame.loc[:, ["frequency", "V(vout)"]].T.values
                    Vout = Vout_freq[1]
                    Freq = Vout_freq[0]
                    print(Vout)
                    plt.subplot(2,1,1)
                    plt.plot(Freq, np.angle(Vout)*180/math.pi)
                    plt.xscale("log")
                    plt.subplot(2,1,2)
                    plt.plot(Freq, np.abs(Vout)/20E-6)
                    plt.xscale("log")
                    plt.yscale("log")
                    plt.show()
                # 读取MOS管静态工作点、交流参数(.log)
                MOS_operating_point = self.MOS_OP_Read(log)

                # print(process.Cal_uCox(MOS_operating_point, a_Length_nm, a_Width_nm))
                print("not in satuation region: ", MOS_operating_point.read_element("region", "others"))
                # 均在饱和区标志位
                if(len(MOS_operating_point.read_element("region", "others")) == 0):
                    sim_result["all_sat"] = 1
                else:
                    sim_result["all_sat"] = 0
                # print(sim_result["all_sat"])
                try:
                    sim_result["P0"] = float(MOS_operating_point.read_element("others", "P0"))
                    sim_result["ro"] = 1/float(MOS_operating_point.read_element("m7", "Gm"))
                    sim_result["AvodB"] = float(MOS_operating_point.read_element("others", "AvodB"))
                    sim_result["BW"] = float(MOS_operating_point.read_element("others", "BW"))
                    sim_result["GBW"] = float(MOS_operating_point.read_element("others", "GBW"))
                    PMdeg = -float(MOS_operating_point.read_element("others", "PMdeg"))
                    sim_result["PMdeg"] = PMdeg
                except:
                    sim_result["all_sat"] = 0
                    sim_result["P0"] = 1E9
                    sim_result["ro"] = 1E9
                    sim_result["AvodB"] = 0
                    sim_result["BW"] = 0
                    sim_result["GBW"] = 0
                    sim_result["PMdeg"] = 0
            # print("end analysis:", time.time()-start)
        netlist.reset_netlist()

        num_sim = self.num_sim

        for delfile in [".net", ".raw", ".log", ".op.raw"]:
            filename = SimFilePath + "temp\\"+SimFileName+"_" + str(num_sim) + delfile
            if(os.path.exists(filename)):
                os.remove(filename)

        self.num_sim = num_sim + 1

        print("\033[1;35m", sim_result, "\033[0m")
        print("\033[1;36m", num_sim, "th, cost", time.time()-start, "s", "\033[0m")
        print("------------------------------- end of simulation -------------------------------")
        return sim_result




if __name__ == "__main__":
    LTspiceExec = "D:\\DATA\\Software\\LTspice\\LTspice.exe"
    SimFilePath = "C:\\Users\\HJH\\Desktop\\simulator\\based_on_python\\OPAMP3\\simfile\\"
    SimFileName = "opamp"

    my_Length_nm = [720 , 720 , 720  , 720  , 720 , 720  , 720 , 720  , 720 , 540 , 540  , 720 , 720 , 720, 720, 720 , 720 , 720, 720]
    my_Width_nm  = [3600, 3600, 10800, 10800, 2880, 42350, 5500, 80000, 5200, 1683, 20197, 1683, 1683, 720, 720, 5120, 5120, 720, 720]
    my_R0 = 30500
    my_Cc = 1.3E-12

    simulator = Simulator(SimFilePath=SimFilePath, 
                          SimFileName=SimFileName, 
                          LTspiceExec=LTspiceExec, 
                          Model="smic18")
    
    simulator.Simulator(my_Width_nm, my_Length_nm, my_R0, my_Cc)