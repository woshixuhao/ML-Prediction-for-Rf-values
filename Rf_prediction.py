'''
The code for 'High-throughput discovery of chemical structure-polarity relationships combining automation and machine learning techniques'
by Hao Xu
revised at 2022.7.6
'''
from sklearn import linear_model
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import argparse
import  os
import pymysql
import pandas as pd
from rdkit import Chem
import numpy as np
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Draw
import mordred.CPSA as CPSA
import mordred
from xgboost.sklearn import XGBClassifier,XGBRegressor
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math
from rdkit.Chem import MACCSkeys
from PIL import Image
import torch.nn as nn
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
import time
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

warnings.filterwarnings("ignore")

font1 = {'family': 'Arial',
         'weight': 'normal',
         #"style": 'italic',
         'size': 14,
         }
font1egend = {'family': 'Arial',
         'weight': 'normal',
         #"style": 'italic',
         'size': 5,
         }
Eluent_smiles=['CCCCCC','CC(OCC)=O','C(Cl)Cl','CO','CCOCC']

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default=os.getcwd()+'\TLC_dataset.xlsx', help='path of download dataset')
    parser.add_argument('--dipole_path', type=str, default=os.getcwd() + '\dipole_moment_info.xlsx',
                        help='path of dipole file')
    parser.add_argument('--data_range', type=int, default=4944, help='utilized data range,robot:4114,manual:4458,new:4944')
    parser.add_argument('--automatic_divide', type=bool, default=False, help='automatically divide dataset by 80% train,10% validate and 10% test')
    parser.add_argument('--choose_total', type=int, default=387, help='train total num,robot:387,manual:530')
    parser.add_argument('--choose_train', type=int, default=308, help='train num,robot:387,manual:530')
    parser.add_argument('--choose_validate', type=int, default=38, help='validate num')
    parser.add_argument('--choose_test', type=int, default=38, help='test num')
    parser.add_argument('--seed', type=int, default=324, help='random seed for split dataset')
    parser.add_argument('--torch_seed', type=int, default=324, help='random seed for torch')
    parser.add_argument('--add_dipole', type=bool, default=True, help='add dipole into dataset')
    parser.add_argument('--add_molecular_descriptors', type=bool, default=True, help='add molecular_descriptors (MW、TPSA、NROTB、HBA、HBD、LogP into dataset')
    parser.add_argument('--add_MACCkeys', type=bool, default=True,help='add MACCSkeys into dataset')
    parser.add_argument('--add_eluent_matrix', type=bool, default=True,help='add eluent matrix into dataset')
    parser.add_argument('--test_mode', type=str, default='robot', help='manual data or robot data or fix, costum test data')
    parser.add_argument('--use_model', type=str, default='Ensemble',help='the utilized model (XGB,LGB,ANN,RF,Ensemble,Bayesian)')
    parser.add_argument('--download_data', type=bool, default=False, help='use local dataset or download from dataset')
    parser.add_argument('--use_sigmoid', type=bool, default=True, help='use sigmoid')
    parser.add_argument('--shuffle_array', type=bool, default=True, help='shuffle_array')
    parser.add_argument('--characterization_mode', type=str, default='standard',
                        help='the characterization mode for the dataset, including standard, precise_TPSA, no_multi')

    #---------------parapmeters for plot---------------------
    parser.add_argument('--plot_col_num', type=int, default=4, help='The col_num in plot')
    parser.add_argument('--plot_row_num', type=int, default=4, help='The row_num in plot')
    parser.add_argument('--plot_importance_num', type=int, default=10, help='The max importance num in plot')
    #--------------parameters For LGB-------------------
    parser.add_argument('--LGB_max_depth', type=int, default=5, help='max_depth for LGB')
    parser.add_argument('--LGB_num_leaves', type=int, default=25, help='num_leaves for LGB')
    parser.add_argument('--LGB_learning_rate', type=float, default=0.007, help='learning_rate for LGB')
    parser.add_argument('--LGB_n_estimators', type=int, default=1000, help='n_estimators for LGB')
    parser.add_argument('--LGB_early_stopping_rounds', type=int, default=200, help='early_stopping_rounds for LGB')

    #---------------parameters for XGB-----------------------
    parser.add_argument('--XGB_n_estimators', type=int, default=200, help='n_estimators for XGB')
    parser.add_argument('--XGB_max_depth', type=int, default=3, help='max_depth for XGB')
    parser.add_argument('--XGB_learning_rate', type=float, default=0.1, help='learning_rate for XGB')

    #---------------parameters for RF------------------------
    parser.add_argument('--RF_n_estimators', type=int, default=1000, help='n_estimators for RF')
    parser.add_argument('--RF_random_state', type=int, default=1, help='random_state for RF')
    parser.add_argument('--RF_n_jobs', type=int, default=1, help='n_jobs for RF')

    #--------------parameters for ANN-----------------------
    parser.add_argument('--NN_hidden_neuron', type=int, default=128, help='hidden neurons for NN')
    parser.add_argument('--NN_optimizer', type=str, default='Adam', help='optimizer for NN (Adam,SGD,RMSprop)')
    parser.add_argument('--NN_lr', type=float, default=0.005, help='learning rate for NN')
    parser.add_argument('--NN_model_save_location', type=str, default=os.getcwd()+'\model_save_NN', help='learning rate for NN')
    parser.add_argument('--NN_max_epoch', type=int, default=5000, help='max training epoch for NN')
    parser.add_argument('--NN_add_sigmoid', type=bool, default=True, help='whether add sigmoid in NN')
    parser.add_argument('--NN_add_PINN', type=bool, default=False, help='whether add PINN in NN')
    parser.add_argument('--NN_epi', type=float, default=100.0, help='The coef of PINN Loss in NN')



    config = parser.parse_args()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return config

class Dataset_process():
    '''
    For processing the data and split the dataset
    '''
    def __init__(self,config):
        super(Dataset_process, self).__init__()
        self.file_path=config.file_path
        self.dipole_path=config.dipole_path
        self.data_range=config.data_range
        self.choose_train=config.choose_train
        self.choose_validate=config.choose_validate
        self.choose_test=config.choose_test
        self.automatic_divide=config.automatic_divide
        self.seed=config.seed
        self.add_dipole=config.add_dipole
        self.add_molecular_descriptors=config.add_molecular_descriptors
        self.add_eluent_matrix = config.add_eluent_matrix
        self.add_MACCkeys=config.add_MACCkeys
        self.test_mode=config.test_mode
        self.download_data = config.download_data
        self.shuffle_array=config.shuffle_array
        self.costum_array=config.costum_array
        self.characterization_mode=config.characterization_mode

    def download_dataset(self,print_info=True):
        '''
        Download the dataset from mysql dataset
        :param print_info: whether print the download information
        :return: None
        '''
        dbconn = pymysql.connect(
            host='bj-cdb-k8stylt6.sql.tencentcdb.com',
            port=60474,
            user='xuhao',
            password='xuhao1101',
            database='TLC',
            charset='utf8',
        )

        # sql语句
        sqlcmd = "select * from tb_TLC"

        # 利用pandas 模块导入mysql数据
        a = pd.read_sql(sqlcmd, dbconn)

        a.to_excel(self.file_path)
        if print_info==True:
            print(f'Dataset has been downloaded, the file path is :{self.file_path}')

    def get_descriptor(self,smiles,ratio):
        compound_mol = Chem.MolFromSmiles(smiles)
        descriptor=[]
        descriptor.append(Descriptors.ExactMolWt(compound_mol))
        descriptor.append(Chem.rdMolDescriptors.CalcTPSA(compound_mol))
        descriptor.append(Descriptors.NumRotatableBonds(compound_mol))  # Number of rotable bonds
        descriptor.append(Descriptors.NumHDonors(compound_mol))  # Number of H bond donors
        descriptor.append(Descriptors.NumHAcceptors(compound_mol)) # Number of H bond acceptors
        descriptor.append(Descriptors.MolLogP(compound_mol)) # LogP
        descriptor=np.array(descriptor)*ratio
        return descriptor

    def get_eluent_descriptor(self,eluent_array):
        eluent=eluent_array
        des = np.zeros([6,])
        for i in range(eluent.shape[0]):
            if eluent[i] != 0:
                e_descriptors = Dataset_process.get_descriptor(self, Eluent_smiles[i], eluent[i])
                des+=e_descriptors
        return des

    def get_3D_conformer(self):
        data_range = self.data_range
        if self.download_data == True:
            Dataset_process.download_dataset(self)
        entire_info = (pd.read_excel(self.file_path, index_col=None, na_values=['NA']).fillna(-1))
        compound_info = entire_info.values[0:data_range]
        compound_ID = compound_info[:, 2]
        compound_smile = compound_info[:, 3]


        # 转变为mol并变成167维向量
        compound_mol = compound_smile.copy()

        use_index = 0
        for i in tqdm(range(len(compound_smile))):
            compound_mol[i] = Chem.MolFromSmiles(compound_smile[i])
            try:
                Finger = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(compound_smile[i]))
            except Exception as e:
                print(f'the compound {compound_ID[i]} has no MACCkeys.')
                continue
            mol = AllChem.MolFromSmiles(compound_smile[i])
            new_mol = Chem.AddHs(mol)
            res = AllChem.EmbedMultipleConfs(new_mol)
            ### MMFF generates multiple conformations
            res = AllChem.MMFFOptimizeMoleculeConfs(new_mol)
            new_mol = Chem.RemoveHs(new_mol)
            Chem.MolToMolFile(new_mol, f'3D_conform/data_{i}.mol')

    def create_dataset(self,data_array,choose_num,compound_ID,dipole_ID,compound_Rf,compound_finger,compound_eluent,dipole_moment,
                       compound_MolWt,compound_TPSA,compound_nRotB,compound_HBD,compound_HBA,compound_LogP):
        '''
        create training/validate/test dataset
        add or not the molecular_descriptors and dipole moments can be controlled
        '''
        y = []
        database_finger = np.zeros([1, 167])
        database_eluent = np.zeros([1, 6])
        database_dipole = np.zeros([1, 1])
        database_descriptor = np.zeros([1, 6])
        for i in range(choose_num):
            index = int(data_array[i])
            ID_loc = np.where(compound_ID == index)[0]
            dipole_loc = np.where(dipole_ID == index)[0]
            for j in ID_loc:
                y.append([compound_Rf[j]])
                database_finger=np.vstack((database_finger,compound_finger[j]))
                database_eluent=np.vstack((database_eluent,compound_eluent[j]))
                if self.add_dipole==True:
                    database_dipole=np.vstack((database_dipole,dipole_moment[dipole_loc]))
                database_descriptor=np.vstack((database_descriptor,np.array([compound_MolWt[j],compound_TPSA[j],compound_nRotB[j],compound_HBD[j],compound_HBA[j],compound_LogP[j]]).reshape([1,6])))

        if self.add_MACCkeys==True:
            X=database_finger.copy()
            X=np.hstack((X,database_eluent))
        else:
            X=database_eluent.copy()

        if self.add_dipole==True:
            X=np.hstack((X,database_dipole))

        if self.add_molecular_descriptors == True:
            X =np.hstack((X,database_descriptor))


        X = np.delete(X, [0], axis=0)
        y = np.array(y)
        return X,y

    def delete_invalid(self,database, h):
        '''
        delete invalid data which is filled with -1 when reading the dataset
        '''
        delete_row_h = np.where(h == -1)[0]
        if delete_row_h.size > 0:
            database = np.delete(database, delete_row_h, axis=0)
            h = np.delete(h, delete_row_h, axis=0)

        delete_row_data = np.where(database == -1)[0]
        if delete_row_data.size > 0:
            database = np.delete(database, delete_row_data, axis=0)
            h = np.delete(h, delete_row_data, axis=0)
        return database,h

    def plot_compound(self,target_ID=-1):
        data_range = self.data_range
        if self.download_data == True:
            Dataset_process.download_dataset(self)
        entire_info = (pd.read_excel(self.file_path, index_col=None, na_values=['NA']).fillna(-1))
        compound_info = entire_info.values[0:data_range]
        compound_ID = compound_info[:, 2]
        compound_smile = compound_info[:, 3]
        compound_name=compound_info[:,10]
        compound_list = np.unique(compound_ID)
        compound_num = compound_list.shape[0]
        # data_array = compound_list.copy()
        # np.random.seed(self.seed)
        # np.random.shuffle(data_array)

        #----------------单个画图-----------------
        # index = target_ID
        # ID_loc = np.where(compound_ID == index)[0][0]
        # smile=compound_smile[ID_loc]
        # mol= Chem.MolFromSmiles(smile)
        # smiles_pic = Draw.MolToImage(mol, size=(500, 500),dpi=300, kekulize=True)
        # plt.figure(20,figsize=(0.5,0.5),dpi=300)
        # plt.imshow(smiles_pic)
        # plt.axis('off')
        # plt.savefig(f'fig_save/compound_{index}.tiff',dpi=300)
        # plt.savefig(f'fig_save/compound_{index}.pdf', dpi=300)
        # plt.show()


        #------------总体画图-----------------
        if target_ID==-1:
            plt.figure(10,figsize=(7,10),dpi=300)
            num=0
            for i in range(350,384):
                index=compound_list[i]
                ID_loc = np.where(compound_ID == index)[0][0]
                smile=compound_smile[ID_loc]
                mol= Chem.MolFromSmiles(smile)
                smiles_pic = Draw.MolToImage(mol, size=(200, 100), kekulize=True)
                plt.subplot(10,7,num+1)
                #plt.title(index)
                plt.imshow(smiles_pic)
                plt.axis('off')
                num+=1
            plt.savefig(f'fig_save/new/compound_{350}~{384}.tiff',dpi=300)
            plt.savefig(f'fig_save/new/compound_{350}~{384}.pdf',dpi=300)
            plt.show()

    def split_dataset(self):
        '''
        split the dataset according to the train/validate/test num
        :return: X_train,y_train,X_validate,y_validate,X_test,y_test,data_array(shuffled compounds)
        '''
        data_range=self.data_range
        if self.download_data==True:
            Dataset_process.download_dataset(self)
        entire_info = (pd.read_excel(self.file_path, index_col=None, na_values=['NA']).fillna(-1))
        compound_info=entire_info.values[0:data_range]
        compound_ID = compound_info[:, 2]
        compound_smile = compound_info[:, 3]
        compound_Rf = compound_info[:, 9]
        eluent = np.array(compound_info[:, 4:9],dtype=np.float32)
        compound_eluent=[]
        for j in range(eluent.shape[0]):
            des=Dataset_process.get_eluent_descriptor(self,eluent[j])
            compound_eluent.append(des.tolist())
        compound_eluent=np.array(compound_eluent)


        if self.add_eluent_matrix==False:
            Eluent_PE = compound_info[:, 4].copy()
            Eluent_EA = compound_info[:, 4].copy()
            for i in range(len(compound_eluent)):
                a = int(compound_eluent[i].split('-', )[1].split('_', )[1])
                b = int(compound_eluent[i].split('-', )[0].split('_', )[1])
                Eluent_PE[i] = b
                Eluent_EA[i] = a
                compound_eluent[i] = a / (a + b)

        # 转变为mol并变成167维向量
        compound_mol = compound_smile.copy()
        compound_finger = np.zeros([len(compound_smile), 167])
        compound_MolWt = np.zeros([len(compound_smile), 1])
        compound_TPSA = np.zeros([len(compound_smile), 1])
        compound_nRotB = np.zeros([len(compound_smile), 1])
        compound_HBD = np.zeros([len(compound_smile), 1])
        compound_HBA = np.zeros([len(compound_smile), 1])
        compound_LogP = np.zeros([len(compound_smile), 1])
        compound_ID_new = np.zeros([len(compound_smile), 1])
        compound_Rf_new = np.zeros([len(compound_smile), 1])
        compound_eluent_new = np.zeros([len(compound_smile), 6])

        use_index=0
        for i in tqdm(range(len(compound_smile))):
            compound_mol[i] = Chem.MolFromSmiles(compound_smile[i])
            try:
                Finger = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(compound_smile[i]))
            except Exception as e:
                print(f'the compound {compound_ID[i]} has no MACCkeys.')
                continue
            fingerprint = np.array([x for x in Finger])
            compound_finger[use_index] = fingerprint
            compound_MolWt[use_index] = Descriptors.ExactMolWt(compound_mol[i])
            if self.characterization_mode=='precise_TPSA':
                mol_conform=Chem.MolFromMolFile(f"3D_conform/data_{i}.mol")
                compound_TPSA[use_index] = mordred.CPSA.TPSA()(mol_conform)
            else:
                compound_TPSA[use_index] = Chem.rdMolDescriptors.CalcTPSA(compound_mol[i])
            compound_nRotB[use_index] = Descriptors.NumRotatableBonds(compound_mol[i])  # Number of rotable bonds
            compound_HBD[use_index] = Descriptors.NumHDonors(compound_mol[i])  # Number of H bond donors
            compound_HBA[use_index] = Descriptors.NumHAcceptors(compound_mol[i])  # Number of H bond acceptors
            compound_LogP[use_index] = Descriptors.MolLogP(compound_mol[i])  # LogP
            compound_ID_new[use_index]=compound_ID[i]
            compound_Rf_new[use_index]=compound_Rf[i]
            compound_eluent_new[use_index]=compound_eluent[i]
            use_index+=1


        compound_ID=compound_ID_new[0:use_index]
        compound_Rf=compound_Rf_new[0:use_index].reshape(compound_ID.shape[0],)
        compound_finger=compound_finger[0:use_index]
        compound_eluent=compound_eluent_new[0:use_index]
        compound_MolWt=compound_MolWt[0:use_index]
        compound_TPSA=compound_TPSA[0:use_index]
        compound_nRotB=compound_nRotB[0:use_index]
        compound_HBD=compound_HBD[0:use_index]
        compound_HBA=compound_HBA[0:use_index]
        compound_LogP=compound_LogP[0:use_index]

        # 读取偶极矩文件
        if self.add_dipole==True:
            dipole_info = (pd.read_excel(self.dipole_path, index_col=None, na_values=['NA']).fillna(-1)).values
            dipole_ID = dipole_info[:, 0]
            dipole_moment = dipole_info[:, 11]
        else:
            dipole_ID = None
            dipole_moment = None

        # 计算化合物的个数
        compound_list = np.unique(compound_ID)
        compound_num = compound_list.shape[0]
        # print(compound_num)
        if self.automatic_divide==True:
            self.choose_train=math.floor(0.8*compound_num)
            self.choose_validate=math.floor(0.1*compound_num)
            self.choose_test = math.floor(0.1 * compound_num)
        # print(self.choose_train,self.choose_validate,self.choose_test)
        if self.choose_train+self.choose_validate+self.choose_test>compound_num:
            raise ValueError(f'Out of compound num, which is {compound_num}')
        data_array = compound_list.copy()
        if self.shuffle_array==True:
            np.random.seed(self.seed)
            np.random.shuffle(data_array)

        X_train,y_train=Dataset_process.create_dataset(self,data_array[0:self.choose_train],self.choose_train,compound_ID, dipole_ID, compound_Rf, compound_finger,
                       compound_eluent, dipole_moment,compound_MolWt, compound_TPSA, compound_nRotB, compound_HBD, compound_HBA, compound_LogP)
        X_validate, y_validate = Dataset_process.create_dataset(self, data_array[self.choose_train:self.choose_train+self.choose_validate], self.choose_validate,
                                                          compound_ID, dipole_ID, compound_Rf, compound_finger,
                                                          compound_eluent, dipole_moment, compound_MolWt, compound_TPSA,
                                                          compound_nRotB, compound_HBD, compound_HBA, compound_LogP)
        if self.test_mode=='robot':
            X_test, y_test=Dataset_process.create_dataset(self, data_array[self.choose_train+self.choose_validate:self.choose_train+self.choose_validate+self.choose_test], self.choose_test,
                                                          compound_ID, dipole_ID, compound_Rf, compound_finger,
                                                          compound_eluent, dipole_moment, compound_MolWt, compound_TPSA,
                                                          compound_nRotB, compound_HBD, compound_HBA, compound_LogP)

        elif self.test_mode=='fix':
            X_test, y_test=Dataset_process.create_dataset(self, data_array[-self.choose_test-1:-1], self.choose_test,
                                                          compound_ID, dipole_ID, compound_Rf, compound_finger,
                                                          compound_eluent, dipole_moment, compound_MolWt, compound_TPSA,
                                                          compound_nRotB, compound_HBD, compound_HBA, compound_LogP)

        elif self.test_mode=='costum':
            X_test, y_test = Dataset_process.create_dataset(self, self.costum_array,
                                                            len(self.costum_array),
                                                            compound_ID, dipole_ID, compound_Rf, compound_finger,
                                                            compound_eluent, dipole_moment, compound_MolWt,
                                                            compound_TPSA,
                                                            compound_nRotB, compound_HBD, compound_HBA, compound_LogP)





        X_train,y_train=Dataset_process.delete_invalid(self,X_train,y_train)
        X_validate, y_validate = Dataset_process.delete_invalid(self, X_validate, y_validate)
        X_test,y_test=Dataset_process.delete_invalid(self, X_test, y_test)


        return X_train,y_train,X_validate,y_validate,X_test,y_test,data_array

    def split_dataset_all(self):
        '''
        split the dataset according to the TLC_num
        :return: X
        '''
        data_range = self.data_range
        if self.download_data == True:
            Dataset_process.download_dataset(self)
        entire_info = (pd.read_excel(self.file_path, index_col=None, na_values=['NA']).fillna(-1))
        compound_info = entire_info.values[0:data_range]
        compound_ID = compound_info[:, 2]
        compound_smile = compound_info[:, 3]
        compound_Rf = compound_info[:, 9]
        eluent = np.array(compound_info[:, 4:9], dtype=np.float32)
        compound_eluent = []
        for j in range(eluent.shape[0]):
            des = Dataset_process.get_eluent_descriptor(self, eluent[j])
            compound_eluent.append(des.tolist())
        compound_eluent = np.array(compound_eluent)
        if self.add_eluent_matrix == False:
            Eluent_PE = compound_info[:, 4].copy()
            Eluent_EA = compound_info[:, 4].copy()
            for i in range(len(compound_eluent)):
                a = int(compound_eluent[i].split('-', )[1].split('_', )[1])
                b = int(compound_eluent[i].split('-', )[0].split('_', )[1])
                Eluent_PE[i] = b
                Eluent_EA[i] = a
                compound_eluent[i] = a / (a + b)

        # 转变为mol并变成167维向量
        compound_mol = compound_smile.copy()
        compound_finger = np.zeros([len(compound_smile), 167])
        compound_MolWt = np.zeros([len(compound_smile), 1])
        compound_TPSA = np.zeros([len(compound_smile), 1])
        compound_nRotB = np.zeros([len(compound_smile), 1])
        compound_HBD = np.zeros([len(compound_smile), 1])
        compound_HBA = np.zeros([len(compound_smile), 1])
        compound_LogP = np.zeros([len(compound_smile), 1])
        compound_ID_new = np.zeros([len(compound_smile), 1])
        compound_Rf_new = np.zeros([len(compound_smile), 1])
        compound_eluent_new = np.zeros([len(compound_smile), 6])

        use_index = 0
        for i in range(len(compound_smile)):
            compound_mol[i] = Chem.MolFromSmiles(compound_smile[i])
            try:
                Finger = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(compound_smile[i]))
            except Exception as e:
                print(f'the compound {compound_ID[i]} has no MACCkeys.')
                continue
            fingerprint = np.array([x for x in Finger])
            compound_finger[use_index] = fingerprint
            compound_MolWt[use_index] = Descriptors.ExactMolWt(compound_mol[i])
            if self.characterization_mode=='precise_TPSA':
                mol_conform=Chem.MolFromMolFile(f"3D_conform/data_{i}.mol")
                compound_TPSA[use_index] = mordred.CPSA.TPSA()(mol_conform)
            else:
                compound_TPSA[use_index] = Chem.rdMolDescriptors.CalcTPSA(compound_mol[i])
            compound_nRotB[use_index] = Descriptors.NumRotatableBonds(compound_mol[i])  # Number of rotable bonds
            compound_HBD[use_index] = Descriptors.NumHDonors(compound_mol[i])  # Number of H bond donors
            compound_HBA[use_index] = Descriptors.NumHAcceptors(compound_mol[i])  # Number of H bond acceptors
            compound_LogP[use_index] = Descriptors.MolLogP(compound_mol[i])  # LogP
            compound_ID_new[use_index] = compound_ID[i]
            compound_Rf_new[use_index] = compound_Rf[i]
            compound_eluent_new[use_index] = compound_eluent[i]
            use_index += 1

        compound_ID = compound_ID_new[0:use_index]
        compound_Rf = compound_Rf_new[0:use_index].reshape(compound_ID.shape[0], )
        compound_finger = compound_finger[0:use_index]
        compound_eluent = compound_eluent_new[0:use_index]
        compound_MolWt = compound_MolWt[0:use_index]
        compound_TPSA = compound_TPSA[0:use_index]
        compound_nRotB = compound_nRotB[0:use_index]
        compound_HBD = compound_HBD[0:use_index]
        compound_HBA = compound_HBA[0:use_index]
        compound_LogP = compound_LogP[0:use_index]
        # 读取偶极矩文件
        if self.add_dipole == True:
            dipole_info = (pd.read_excel(self.dipole_path, index_col=None, na_values=['NA']).fillna(-1)).values
            dipole_ID = dipole_info[:, 0]
            dipole_moment = dipole_info[:, 11]
        else:
            dipole_ID = None
            dipole_moment = None

        y = []
        ID=[]
        database_finger = np.zeros([1, 167])
        database_eluent = np.zeros([1, 6])
        database_dipole = np.zeros([1, 1])
        database_descriptor = np.zeros([1, 6])
        for i in range(compound_finger.shape[0]):
            dipole_loc = np.where(dipole_ID == compound_ID[i])[0]
            y.append([compound_Rf[i]])
            ID.append([compound_ID[i]])
            database_finger = np.vstack((database_finger, compound_finger[i]))
            database_eluent = np.vstack((database_eluent, compound_eluent[i]))
            if self.add_dipole == True:
                database_dipole = np.vstack((database_dipole, dipole_moment[dipole_loc]))
            database_descriptor = np.vstack((database_descriptor, np.array(
                [compound_MolWt[i], compound_TPSA[i], compound_nRotB[i], compound_HBD[i], compound_HBA[i],
                 compound_LogP[i]]).reshape([1, 6])))

        if self.add_MACCkeys==True:
            X = database_finger.copy()
            X = np.hstack((X, database_eluent))
        else:
            X = database_eluent.copy()
        if self.add_dipole == True:
            X = np.hstack((X, database_dipole))
        if self.add_molecular_descriptors == True:
            X = np.hstack((X, database_descriptor))

        if self.characterization_mode=='no_multi':
            X = np.delete(X, [27,42,46,103], axis=1)
        X = np.delete(X, [0], axis=0)
        y = np.array(y)
        return X,y,ID

class ANN(nn.Module):
    '''
    Construct artificial neural network
    '''
    def __init__(self, in_neuron, hidden_neuron, out_neuron,config):
        super(ANN, self).__init__()
        self.input_layer = nn.Linear(in_neuron, hidden_neuron)
        self.hidden_layer = nn.Linear(hidden_neuron, hidden_neuron)
        self.output_layer = nn.Linear(hidden_neuron, out_neuron)
        self.NN_add_sigmoid=config.NN_add_sigmoid


    def forward(self, x):
        x = self.input_layer(x)
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)
        x = F.leaky_relu(x)
        x = self.hidden_layer(x)
        x = F.leaky_relu(x)
        x = self.output_layer(x)
        if self.NN_add_sigmoid==True:
            x = F.sigmoid(x)
        return x

class Model_ML():
    def __init__(self,config):
        super(Model_ML, self).__init__()
        self.seed=config.seed
        self.torch_seed=config.seed
        self.config=config
        self.data_range=config.data_range
        self.file_path=config.file_path
        self.choose_train = config.choose_train
        self.choose_validate = config.choose_validate
        self.choose_test = config.choose_test
        self.add_dipole = config.add_dipole
        self.add_molecular_descriptors = config.add_molecular_descriptors
        self.add_eluent_matrix=config.add_eluent_matrix
        self.use_sigmoid=config.use_sigmoid

        self.use_model=config.use_model
        self.LGB_max_depth=config.LGB_max_depth
        self.LGB_num_leaves=config.LGB_num_leaves
        self.LGB_learning_rate=config.LGB_learning_rate
        self.LGB_n_estimators=config.LGB_n_estimators
        self.LGB_early_stopping_rounds=config.LGB_early_stopping_rounds

        self.XGB_n_estimators=config.XGB_n_estimators
        self.XGB_max_depth = config.XGB_max_depth
        self.XGB_learning_rate = config.XGB_learning_rate

        self.RF_n_estimators=config.RF_n_estimators
        self.RF_random_state=config.RF_random_state
        self.RF_n_jobs=config.RF_n_jobs

        self.NN_hidden_neuron=config.NN_hidden_neuron
        self.NN_optimizer=config.NN_optimizer
        self.NN_lr= config.NN_lr
        self.NN_model_save_location=config.NN_model_save_location
        self.NN_max_epoch=config.NN_max_epoch
        self.NN_add_PINN=config.NN_add_PINN
        self.NN_epi=config.NN_epi
        self.device=config.device

        self.plot_row_num=config.plot_row_num
        self.plot_col_num=config.plot_col_num
        self.plot_importance_num=config.plot_importance_num

    def train(self,X_train,y_train,X_validate,y_validate):


        '''
        train model using LightGBM,Xgboost,Random forest or ANN
        '''
        print('----------Start Training!--------------')
        torch.manual_seed(self.torch_seed)
        if self.use_model=='LGB':
            if self.use_sigmoid == True:
                y_train = np.clip(y_train, 0.01, 0.99)
                y_validate = np.clip(y_validate, 0.01, 0.99)
                y_train = np.log(y_train / (1 - y_train))
                y_validate = np.log(y_validate / (1 - y_validate))
            model = lgb.LGBMRegressor(objective='regression', max_depth=self.LGB_max_depth,
                                      num_leaves=self.LGB_num_leaves,
                                      learning_rate=self.LGB_learning_rate, n_estimators=self.LGB_n_estimators)
            model.fit(X_train, list(y_train.reshape(y_train.shape[0], )),
                      eval_set=[(X_train, list(y_train.reshape(y_train.shape[0], ))),
                                (X_validate, list(y_validate.reshape(y_validate.shape[0], )))],
                      eval_names=('fit', 'val'), eval_metric='l2', early_stopping_rounds=self.LGB_early_stopping_rounds,
                      verbose=False)
            print('----------LGB Training Finished!--------------')
            return model
        elif self.use_model=='XGB':
            if self.use_sigmoid == True:
                y_train = np.clip(y_train, 0.01, 0.99)
                y_validate = np.clip(y_validate, 0.01, 0.99)
                y_train = np.log(y_train / (1 - y_train))
                y_validate = np.log(y_validate / (1 - y_validate))
            model = XGBRegressor(seed=self.seed,
                                 n_estimators=self.XGB_n_estimators,
                                 max_depth=self.XGB_max_depth,
                                 eval_metric='rmse',
                                 learning_rate=self.XGB_learning_rate,
                                 min_child_weight=1,
                                 subsample=1,
                                 colsample_bytree=1,
                                 colsample_bylevel=1,
                                 gamma=0)

            model.fit(X_train, y_train.reshape(y_train.shape[0]))
            print('----------XGB Training Finished!--------------')
            return model

        elif self.use_model=='RF':
            if self.use_sigmoid == True:
                y_train = np.clip(y_train, 0.01, 0.99)
                y_validate = np.clip(y_validate, 0.01, 0.99)
                y_train = np.log(y_train / (1 - y_train))
                y_validate = np.log(y_validate / (1 - y_validate))
            model = RandomForestRegressor(n_estimators=self.RF_n_estimators,
                                          criterion='mse',
                                          random_state=self.RF_random_state,
                                          n_jobs=self.RF_n_jobs)
            model.fit(X_train, y_train)
            print('----------RF Training Finished!--------------')
            return model

        elif self.use_model=='ANN':
            Net = ANN(X_train.shape[1], self.NN_hidden_neuron, 1, config=self.config).to(self.device)
            X_train = Variable(torch.from_numpy(X_train.astype(np.float32)).to(self.device), requires_grad=True)
            y_train = Variable(torch.from_numpy(y_train.astype(np.float32)).to(self.device))
            X_validate = Variable(torch.from_numpy(X_validate.astype(np.float32)).to(self.device), requires_grad=True)
            y_validate = Variable(torch.from_numpy(y_validate.astype(np.float32)).to(self.device))

            model_name = time.strftime('%m-%d-%H-%M', time.localtime(time.time()))
            dir_name = self.NN_model_save_location + '/' + model_name
            loss_plot = []
            loss_validate_plot = []
            try:
                os.makedirs(dir_name)
            except OSError:
                pass

            if self.NN_optimizer == 'SGD':
                optimizer = torch.optim.SGD(Net.parameters(), lr=self.NN_lr)
            elif self.NN_optimizer == 'RMSprop':
                optimizer = torch.optim.RMSprop(Net.parameters(), lr=self.NN_lr)
            else:
                optimizer = torch.optim.Adam(Net.parameters(), lr=self.NN_lr)

            with open(dir_name + '/' + 'data.txt', 'w') as f:  # 设置文件对象
                for epoch in range(self.NN_max_epoch):
                    optimizer.zero_grad()
                    prediction = Net(X_train)
                    prediction_validate = Net(X_validate)
                    dprediction = \
                    torch.autograd.grad(outputs=prediction[:, 0].sum(), inputs=X_train, create_graph=True)[0]
                    dprediction_deluent = dprediction[:, 167].reshape(X_train.shape[0], 1)
                    dprediction_validate = \
                    torch.autograd.grad(outputs=prediction_validate[:, 0].sum(), inputs=X_validate,
                                        create_graph=True)[0]
                    dprediction_validate_deluent = dprediction_validate[:, 167].reshape(X_validate.shape[0], 1)
                    MSELoss=torch.nn.MSELoss()
                    if self.NN_add_PINN == True:
                        loss = MSELoss(y_train, prediction) + self.NN_epi * (
                            torch.sum(F.relu(-dprediction_deluent))) / X_train.shape[0]
                        loss_validate = MSELoss(y_validate, prediction_validate) + (self.NN_epi * torch.sum(
                            F.relu(-dprediction_validate_deluent))) / X_validate.shape[0]
                    else:
                        loss = MSELoss(y_train, prediction)
                        loss_validate = MSELoss(y_validate, prediction_validate)

                    loss.backward()
                    optimizer.step()
                    if epoch>200:
                        if loss.item() == loss_plot[-1]:
                            if self.NN_optimizer == 'SGD':
                                optimizer = torch.optim.SGD(Net.parameters(), lr=self.NN_lr * 0.2)
                            elif self.NN_optimizer == 'RMSprop':
                                optimizer = torch.optim.RMSprop(Net.parameters(), lr=self.NN_lr * 0.2)
                            else:
                                optimizer = torch.optim.Adam(Net.parameters(), lr=self.NN_lr * 0.2)

                    if (epoch + 1) % 100 == 0:
                        print("iter_num: %d      loss: %.8f    loss_validate: %.8f" % (
                        epoch + 1, loss.item(), loss_validate.item()))
                        f.write("iter_num: %d      loss: %.8f    loss_validate: %.8f \r\n" % (
                        epoch + 1, loss.item(), loss_validate.item()))
                        torch.save(Net.state_dict(), dir_name + '/' + "%d_epoch.pkl" % (epoch+1))
                        loss_plot.append(loss.item())
                        loss_validate_plot.append(loss_validate.item())

                best_epoch=(loss_validate_plot.index(min(loss_validate_plot))+1)*100
                print("The ANN has been trained, the best epoch is %d"%(best_epoch))
                Net.load_state_dict(torch.load(dir_name + '/' + "%d_epoch.pkl" % (best_epoch)))
                Net.eval()

                plt.figure(3)
                plt.plot(loss_plot, marker='x', label='loss')
                plt.plot(loss_validate_plot, c='red', marker='v', label='loss_validate')
                plt.legend()
                plt.savefig(dir_name + '/' + 'loss_pic.png')
                print('----------ANN Training Finished!--------------')
            return Net

        elif self.use_model=='Bayesian':
            if self.use_sigmoid == True:
                y_train = np.clip(y_train, 0.01, 0.99)
                y_validate = np.clip(y_validate, 0.01, 0.99)
                y_train = np.log(y_train / (1 - y_train))
                y_validate = np.log(y_validate / (1 - y_validate))
            clf = linear_model.BayesianRidge()
            clf.fit(X_train, y_train.reshape(y_train.shape[0]))
            return clf

        elif self.use_model=='Ensemble':
            y_train_origin=y_train.copy()
            y_validate_origin = y_validate.copy()
            if self.use_sigmoid == True:
                y_train = np.clip(y_train, 0.01, 0.99)
                y_validate = np.clip(y_validate, 0.01, 0.99)
                y_train = np.log(y_train / (1 - y_train))
                y_validate = np.log(y_validate / (1 - y_validate))
            model_LGB = lgb.LGBMRegressor(objective='regression', max_depth=self.LGB_max_depth,
                                      num_leaves=self.LGB_num_leaves,
                                      learning_rate=self.LGB_learning_rate, n_estimators=self.LGB_n_estimators)
            model_LGB.fit(X_train, list(y_train.reshape(y_train.shape[0], )),
                      eval_set=[(X_train, list(y_train.reshape(y_train.shape[0], ))),
                                (X_validate, list(y_validate.reshape(y_validate.shape[0], )))],
                      eval_names=('fit', 'val'), eval_metric='l2', early_stopping_rounds=self.LGB_early_stopping_rounds,
                      verbose=False)
            model_XGB = XGBRegressor(seed=self.seed,
                                 n_estimators=self.XGB_n_estimators,
                                 max_depth=self.XGB_max_depth,
                                 eval_metric='rmse',
                                 learning_rate=self.XGB_learning_rate,
                                 min_child_weight=1,
                                 subsample=1,
                                 colsample_bytree=1,
                                 colsample_bylevel=1,
                                 gamma=0)

            model_XGB.fit(X_train, y_train.reshape(y_train.shape[0]))
            model_RF = RandomForestRegressor(n_estimators=self.RF_n_estimators,
                                          criterion='mse',
                                          random_state=self.RF_random_state,
                                          n_jobs=self.RF_n_jobs)
            model_RF.fit(X_train, y_train)

            Net = ANN(X_train.shape[1], self.NN_hidden_neuron, 1, config=self.config).to(self.device)
            X_train = Variable(torch.from_numpy(X_train.astype(np.float32)).to(self.device), requires_grad=True)
            y_train = Variable(torch.from_numpy(y_train_origin.astype(np.float32)).to(self.device))
            X_validate = Variable(torch.from_numpy(X_validate.astype(np.float32)).to(self.device), requires_grad=True)
            y_validate = Variable(torch.from_numpy(y_validate_origin.astype(np.float32)).to(self.device))

            model_name = time.strftime('%m-%d-%H-%M', time.localtime(time.time()))
            dir_name = self.NN_model_save_location + '/' + model_name
            loss_plot = []
            loss_validate_plot = []
            try:
                os.makedirs(dir_name)
            except OSError:
                pass

            if self.NN_optimizer == 'SGD':
                optimizer = torch.optim.SGD(Net.parameters(), lr=self.NN_lr)
            elif self.NN_optimizer == 'RMSprop':
                optimizer = torch.optim.RMSprop(Net.parameters(), lr=self.NN_lr)
            else:
                optimizer = torch.optim.Adam(Net.parameters(), lr=self.NN_lr)

            with open(dir_name + '/' + 'data.txt', 'w') as f:  # 设置文件对象
                for epoch in range(self.NN_max_epoch):
                    optimizer.zero_grad()
                    prediction = Net(X_train)
                    prediction_validate = Net(X_validate)
                    MSELoss = torch.nn.MSELoss()
                    if self.NN_add_PINN == True:
                        dprediction = \
                            torch.autograd.grad(outputs=prediction[:, 0].sum(), inputs=X_train, create_graph=True)[0]
                        dprediction_deluent = dprediction[:, 167].reshape(X_train.shape[0], 1)
                        dprediction_validate = \
                            torch.autograd.grad(outputs=prediction_validate[:, 0].sum(), inputs=X_validate,
                                                create_graph=True)[0]
                        dprediction_validate_deluent = dprediction_validate[:, 167].reshape(X_validate.shape[0], 1)
                        loss = MSELoss(y_train, prediction) + self.NN_epi * (
                            torch.sum(F.relu(-dprediction_deluent))) / X_train.shape[0]
                        loss_validate = MSELoss(y_validate, prediction_validate) + (self.NN_epi * torch.sum(
                            F.relu(-dprediction_validate_deluent))) / X_validate.shape[0]
                    else:
                        loss = MSELoss(y_train, prediction)
                        loss_validate = MSELoss(y_validate, prediction_validate)

                    loss.backward()
                    optimizer.step()


                    if (epoch + 1) % 100 == 0:
                        print("iter_num: %d      loss: %.8f    loss_validate: %.8f" % (
                            epoch + 1, loss.item(), loss_validate.item()))
                        f.write("iter_num: %d      loss: %.8f    loss_validate: %.8f \r\n" % (
                            epoch + 1, loss.item(), loss_validate.item()))
                        torch.save(Net.state_dict(), dir_name + '/' + "%d_epoch.pkl" % (epoch + 1))
                        loss_plot.append(loss.item())
                        loss_validate_plot.append(loss_validate.item())

                best_epoch = (loss_validate_plot.index(min(loss_validate_plot)) + 1) * 100
                print("The ANN has been trained, the best epoch is %d" % (best_epoch))
                Net.load_state_dict(torch.load(dir_name + '/' + "%d_epoch.pkl" % (best_epoch)))
                Net.eval()

                plt.figure(100)
                plt.plot(loss_plot, marker='x', label='loss')
                plt.plot(loss_validate_plot, c='red', marker='v', label='loss_validate')
                plt.legend()
                plt.savefig(dir_name + '/' + 'loss_pic.png')
            return model_LGB,model_XGB,model_RF,Net

    def plot_total_variance(self,y_test,y_pred):
        '''
        plot and calculate the MSE, RMSE, MAE and R^2
        '''
        if self.use_model=='ANN':
            y_test=y_test.cpu().data.numpy()

        y_test=y_test.reshape(y_test.shape[0])
        y_pred=y_pred.reshape(y_pred.shape[0])

        # y_plot=np.hstack((y_test.reshape(y_test.shape[0],1),y_pred.reshape(y_pred.shape[0],1)))
        # df=pd.DataFrame(y_plot)
        # df.columns = ['True_value', 'Prediction_value']
        # df['method']=self.use_model
        #print(df)
        #df.to_csv(f"result_save/revised_{self.use_model}_compound.csv")
        #
        ## ------------plot total loss---------------
        # print(df)
        # base_plot = (ggplot(df) +geom_point(aes('True_value', 'Prediction_value'),alpha=0.3,color="blue",size=4)
        #               +geom_line(aes('True_value','True_value'),linetype='--',size=1)+ggtitle(self.use_model)+
        #               xlab('Observed Yield')+ylab('Prediction Yield'))#+theme(axis_text=element_text(size=16)))
        # print(base_plot)

        MSE = np.sum(np.abs(y_test - y_pred)**2) /y_test.shape[0]
        RMSE=np.sqrt(MSE)
        MAE = np.sum(np.abs(y_test - y_pred)) / y_test.shape[0]
        R_square=1-(((y_test-y_pred)**2).sum()/((y_test-y_test.mean())**2).sum())
        #print(f"MSE is {MSE}, RMSE is {RMSE}, MAE is {MAE}, R_square is {R_square}")
        return MSE,RMSE,MAE,R_square

    def plot_predict_polarity(self,X_test,y_test,data_array,model):
        '''
        plot prediction for each compound in the test dataset
        '''
        plt.style.use('ggplot')
        data_range=self.data_range
        if self.use_model=='ANN':
            X_test=X_test.cpu().data.numpy()
            y_test = y_test.cpu().data.numpy()

        Dataset_process.download_dataset(self,print_info=False)
        entire_info = (pd.read_excel(self.file_path, index_col=None, na_values=['NA']).fillna(-1))
        compound_info=entire_info.values[0:data_range]
        compound_ID = compound_info[:, 2]
        compound_smile = compound_info[:, 3]
        compound_Rf = compound_info[:, 9]
        compound_eluent = np.array(compound_info[:, 4:9], dtype=np.float32)


        N = 8
        x_origin = np.array([[0,1,0,0,0],[0.333333,0.666667,0,0,0],[0.5,0.5,0,0,0],
                      [0.75,0.25,0,0,0],[0.833333,0.166667,0,0,0],[0.952381,0.047619,0,0,0],
                      [0.980392,0.019608,0,0,0],[1,	0,	0,	0,0]],dtype=np.float32)
        x_ME_origin=np.array([[0,0,1,0,0],[0,0,0.990099,0.009901,0],[0,0,0.980392,0.019608,0],
                      [0,0,0.967742,0.032258,0],[0,0,0.952381,0.047619,0],[0,0,0.909091,0.090909,0]],dtype=np.float32)
        x_Et_origin=np.array([[0.66667,0,0,0,0.33333],[0.5,0,0,0,0.5],[0,0,0,0,1]])
        x=[]
        x_ME=[]
        x_Et=[]
        for i in range(x_origin.shape[0]):
            x.append(Dataset_process.get_eluent_descriptor(self,x_origin[i]))
        x=np.array(x)

        for i in range(x_ME_origin.shape[0]):
            x_ME.append(Dataset_process.get_eluent_descriptor(self,x_ME_origin[i]))
        x_ME=np.array(x_ME)

        for i in range(x_Et_origin.shape[0]):
            x_Et.append(Dataset_process.get_eluent_descriptor(self,x_Et_origin[i]))
        x_Et=np.array(x_Et)

        X_test_origin=X_test.copy()
        X_test[:,167:173]=0.0
        unique_rows,inv = np.unique(X_test.astype("<U22"),axis=0,return_inverse=True)

        index = data_array[
               self.choose_train + self.choose_validate:self.choose_train + self.choose_validate + self.choose_test]

        # if self.plot_col_num*self.plot_row_num!=unique_rows.shape[0]:
        #     raise Warning("col_num*row_num should equal to choose_test")

        for j in range(unique_rows.shape[0]):
            database_test = np.zeros([N, unique_rows.shape[1]])
            database_test_ME = np.zeros([6, unique_rows.shape[1]])
            database_test_Et = np.zeros([3, unique_rows.shape[1]])
            index_inv=np.where(inv==j)[0]
            a=len(np.unique(np.array(inv[0:index_inv[0]+1]).astype("<U22"),axis=0))-1
            print(index[a])
            ID_loc = np.where(compound_ID == index[a])[0]
            smiles = compound_smile[ID_loc[0]]
            eluents=compound_eluent[ID_loc]
            Rfs=compound_Rf[ID_loc]


            mol = Chem.MolFromSmiles(smiles)

            for i in range(N):
                database_test[i]=X_test_origin[index_inv[0]]
                database_test[i,167:173]=x[i]
            for i in range(6):
                database_test_ME[i]=X_test_origin[index_inv[0]]
                database_test_ME[i,167:173]=x_ME[i]
            for i in range(3):
                database_test_Et[i]=X_test_origin[index_inv[0]]
                database_test_Et[i,167:173]=x_ME[i]
            if self.use_model=='ANN':
                database_test=Variable(torch.from_numpy(database_test.astype(np.float32)).to(self.device), requires_grad=True)
                y_pred=model(database_test).cpu().data.numpy()
                y_pred =y_pred.reshape(y_pred.shape[0],)

                database_test_ME=Variable(torch.from_numpy(database_test_ME.astype(np.float32)).to(self.device), requires_grad=True)
                y_pred_ME=model(database_test_ME).cpu().data.numpy()
                y_pred_ME =y_pred_ME.reshape(y_pred_ME.shape[0],)

                database_test_Et = Variable(torch.from_numpy(database_test_Et.astype(np.float32)).to(self.device),
                                            requires_grad=True)
                y_pred_Et = model(database_test_Et).cpu().data.numpy()
                y_pred_Et = y_pred_Et.reshape(y_pred_Et.shape[0], )

            elif self.use_model=='Ensemble':
                model_LGB,model_XGB,model_RF,Net=model

                y_pred_LGB = model_LGB.predict(database_test)
                y_pred_ME_LGB = model_LGB.predict(database_test_ME)
                y_pred_Et_LGB = model_LGB.predict(database_test_Et)
                y_pred_XGB = model_XGB.predict(database_test)
                y_pred_ME_XGB = model_XGB.predict(database_test_ME)
                y_pred_Et_XGB = model_XGB.predict(database_test_Et)
                y_pred_RF = model_RF.predict(database_test)
                y_pred_ME_RF = model_RF.predict(database_test_ME)
                y_pred_Et_RF=model_RF.predict(database_test_Et)
                if self.use_sigmoid == True:
                    y_pred_LGB= 1 / (1 + np.exp(-y_pred_LGB))
                    y_pred_ME_LGB = 1 / (1 + np.exp(-y_pred_ME_LGB))
                    y_pred_Et_LGB = 1 / (1 + np.exp(-y_pred_Et_LGB))
                    y_pred_XGB= 1 / (1 + np.exp(-y_pred_XGB))
                    y_pred_ME_XGB = 1 / (1 + np.exp(-y_pred_ME_XGB))
                    y_pred_Et_XGB = 1 / (1 + np.exp(-y_pred_Et_XGB))
                    y_pred_RF= 1 / (1 + np.exp(-y_pred_RF))
                    y_pred_ME_RF = 1 / (1 + np.exp(-y_pred_ME_RF))
                    y_pred_Et_RF = 1 / (1 + np.exp(-y_pred_Et_RF))


                database_test = Variable(torch.from_numpy(database_test.astype(np.float32)).to(self.device),
                                         requires_grad=True)
                y_pred = Net(database_test).cpu().data.numpy()
                y_pred_NN = y_pred.reshape(y_pred.shape[0], )

                database_test_ME = Variable(torch.from_numpy(database_test_ME.astype(np.float32)).to(self.device),
                                            requires_grad=True)
                y_pred_ME = Net(database_test_ME).cpu().data.numpy()
                y_pred_ME_NN = y_pred_ME.reshape(y_pred_ME.shape[0], )

                database_test_Et = Variable(torch.from_numpy(database_test_Et.astype(np.float32)).to(self.device),
                                            requires_grad=True)
                y_pred_Et = Net(database_test_Et).cpu().data.numpy()
                y_pred_Et_NN = y_pred_Et.reshape(y_pred_Et.shape[0], )

                y_pred_ME=(0.2*y_pred_ME_LGB+0.2*y_pred_ME_XGB+0.2*y_pred_ME_RF+0.4*y_pred_ME_NN)
                y_pred_Et=(0.2*y_pred_Et_LGB+0.2*y_pred_Et_XGB+0.2*y_pred_Et_RF+0.4*y_pred_Et_NN)
                y_pred=(0.2*y_pred_NN+0.2*y_pred_XGB+0.2*y_pred_RF+0.4*y_pred_LGB)


            else:
                y_pred=model.predict(database_test)
                y_pred_ME = model.predict(database_test_ME)
                y_pred_Et = model.predict(database_test_Et)
                if self.use_sigmoid==True:
                    y_pred=1/(1+np.exp(-y_pred))
                    y_pred_ME = 1 / (1 + np.exp(-y_pred_ME))
                    y_pred_Et = 1 / (1 + np.exp(-y_pred_Et))



            EA_plot=[]
            y_EA_plot=[]
            Me_plot=[]
            y_ME_plot=[]
            Et_plot=[]
            y_Et_plot=[]
            for k in range(eluents.shape[0]):
                if eluents[k,1]+eluents[k,0]+eluents[k,4]==0:
                    Me_plot.append(np.log(np.array(eluents[k,3] + 1, dtype=np.float32)))
                    y_ME_plot.append(Rfs[k])
                if eluents[k, 2] + eluents[k, 3] + eluents[k, 4] == 0:
                    EA_plot.append(np.log(np.array(eluents[k, 1] + 1, dtype=np.float32)))
                    y_EA_plot.append(Rfs[k])
                if eluents[k, 1] + eluents[k,2] + eluents[k, 3] == 0 and eluents[k,4]!=0:
                    Et_plot.append(np.log(np.array(eluents[k, 4] + 1, dtype=np.float32)))
                    y_Et_plot.append(Rfs[k])

            #plt.style.use('ggplot')
            plt.figure(1,figsize=(2,2),dpi=300)
            ax=plt.subplot(1,1,1)
            #plt.scatter(Me_plot,y_ME_plot,c='pink',marker='v',s=200)
            plt.scatter(EA_plot,y_EA_plot,c='red',marker='^',label='True $R_f$ values',s=30,zorder=1)
            plt.plot(np.log(x_origin[:,1] + 1), y_pred,marker='x',markersize=5,linewidth=1, label='predicted $R_f$ curve',color='blue',zorder=2)
            #plt.plot(np.log(x_ME[:,3] + 1), y_pred_ME,linewidth=3, label='predict Rf curve',color='pink')
            # plt.scatter(np.log(np.array(X_test_origin[index_inv][:,167] + 1, dtype=np.float32)), y_test[index_inv], c='red',
            #             marker='^',s=200, label='True Rf')
            #plt.plot(np.log(x[:,0] + 1), y_pred,marker='o',markersize=10,linewidth=3, label='predict Rf curve')

            # plt.xlabel('Log EA ratio',font1)
            # plt.ylabel('Rf',font1)
            xmajorLocator = MultipleLocator(0.2)
            ax.xaxis.set_major_locator(xmajorLocator)
            plt.yticks(fontproperties='Arial', size=7)
            plt.xticks(fontproperties='Arial', size=7)
            plt.legend(loc='lower right', prop=font1egend)
            plt.ylim(-0.1, 1.1)

            plt.savefig(f'PPT_fig/PE_EA_{int(index[a])}.tiff',
                        bbox_inches='tight', dpi=300)
            plt.savefig(f'PPT_fig/PE_EA_{int(index[a])}.pdf',
                        bbox_inches='tight', dpi=300)
            plt.cla()
            # plt.tight_layout()

            plt.figure(2, figsize=(2, 2), dpi=300)
            smiles_pic = Draw.MolToImage(mol, size=(500, 500), kekulize=True)
            plt.axis('off')
            plt.imshow(smiles_pic)
            plt.savefig(f'PPT_fig/PE_EA_mol_{int(index[a])}.png',
                       bbox_inches='tight', dpi=300)
            plt.savefig(f'PPT_fig/PE_EA_mol_{int(index[a])}.pdf',
                        bbox_inches='tight', dpi=300)
            plt.cla()
            # plt.legend()

            plt.figure(3,figsize=(2,2),dpi=300)
            ax=plt.subplot(1,1,1)
            plt.scatter(Me_plot, y_ME_plot, c='green', marker='^',label='True $R_f$ values', s=30)
            #plt.scatter(EA_plot, y_EA_plot, c='red', marker='^', s=200)
            #plt.plot(np.log(x[:, 1] + 1), y_pred, linewidth=3, label='predict Rf curve', color='red')
            plt.plot(np.log(x_ME[:, 3] + 1), y_pred_ME, marker='x',markersize=5,linewidth=1, label='predicted $R_f$ curve', color='blue')
            # plt.scatter(np.log(np.array(X_test_origin[index_inv][:,167] + 1, dtype=np.float32)), y_test[index_inv], c='red',
            #             marker='^',s=200, label='True Rf')
            # plt.plot(np.log(x[:,0] + 1), y_pred,marker='o',markersize=10,linewidth=3, label='predict Rf curve')

            # plt.xlabel('Log MeOH ratio',font1)
            # plt.ylabel('Rf',font1)
            plt.legend(loc='lower right', prop=font1egend)
            xmajorLocator = MultipleLocator(0.02)
            ax.xaxis.set_major_locator(xmajorLocator)
            plt.yticks(fontproperties='Arial', size=7)
            plt.xticks(fontproperties='Arial', size=7)
            plt.ylim(-0.1, 1.1)
            plt.savefig("temp_fig.png", dpi=300)
            plt.savefig(f'PPT_fig/DCM_MeOH_{int(index[a])}.tiff',
                        bbox_inches='tight', dpi=300)
            plt.savefig(f'PPT_fig/DCM_MeOH_{int(index[a])}.pdf',
                        bbox_inches='tight', dpi=300)
            plt.cla()



            plt.figure(4,figsize=(2,2),dpi=300)
            ax=plt.subplot(1,1,1)
            plt.scatter(Et_plot, y_Et_plot, c='orange',label='True $R_f$ values', marker='^', s=30)
            # plt.scatter(EA_plot, y_EA_plot, c='red', marker='^', s=200)
            # plt.plot(np.log(x[:, 1] + 1), y_pred, linewidth=3, label='predict Rf curve', color='red')
            plt.plot(np.log(x_Et[:, 4] + 1), y_pred_Et, marker='x', markersize=5, linewidth=1,
                     label='predict $R_f$ curve', color='blue')
            # plt.scatter(np.log(np.array(X_test_origin[index_inv][:,167] + 1, dtype=np.float32)), y_test[index_inv], c='red',
            #             marker='^',s=200, label='True Rf')
            # plt.plot(np.log(x[:,0] + 1), y_pred,marker='o',markersize=10,linewidth=3, label='predict Rf curve')
            xmajorLocator = MultipleLocator(0.1)
            ax.xaxis.set_major_locator(xmajorLocator)
            plt.legend(loc='lower right', prop=font1egend)
            plt.yticks(fontproperties='Arial', size=7)
            plt.xticks(fontproperties='Arial', size=7)
            plt.ylim(-0.1, 1.1)

            plt.savefig(f'PPT_fig/PE_Et2O_{int(index[a])}.tiff',
                        bbox_inches='tight', dpi=300)
            plt.savefig(f'PPT_fig/PE_Et2O_{int(index[a])}.pdf',
                        bbox_inches='tight', dpi=300)
            #plt.show()
            plt.cla()
            print(f'{j}:{smiles}')


    def test(self,X_test,y_test,data_array,model):
        '''
        Get test outcomes
        '''
        if self.use_model=='ANN':
            X_test = Variable(torch.from_numpy(X_test.astype(np.float32)).to(self.device), requires_grad=True)
            y_test = Variable(torch.from_numpy(y_test.astype(np.float32)).to(self.device))
            # X_importance=Variable(torch.zeros([1,X_test.shape[1]]).to(self.device),requires_grad=True)
            # y_importance=torch.autograd.grad(outputs=model(X_importance)[:, 0].sum(), inputs=X_importance, create_graph=True)[0]
            # print(y_importance)
            y_pred=model(X_test).cpu().data.numpy()
        elif self.use_model=='Ensemble':
            model_LGB,model_XGB,model_RF,model_ANN =model
            X_test_ANN = Variable(torch.from_numpy(X_test.astype(np.float32)).to(self.device), requires_grad=True)
            y_pred_ANN = model_ANN(X_test_ANN).cpu().data.numpy()
            y_pred_ANN=y_pred_ANN.reshape(y_pred_ANN.shape[0],)

            y_pred_XGB = model_XGB.predict(X_test)
            if self.use_sigmoid == True:
                y_pred_XGB = 1 / (1 + np.exp(-y_pred_XGB))

            y_pred_LGB = model_LGB.predict(X_test)
            if self.use_sigmoid == True:
                y_pred_LGB = 1 / (1 + np.exp(-y_pred_LGB))

            y_pred_RF = model_RF.predict(X_test)
            if self.use_sigmoid == True:
                y_pred_RF = 1 / (1 + np.exp(-y_pred_RF))

            #print(y_pred_LGB.shape,y_pred_XGB.shape,y_pred_ANN.shape)
            self.use_model='Ensemble'
            y_pred=(0.2*y_pred_LGB+0.2*y_pred_XGB+0.2*y_pred_RF+0.4*y_pred_ANN)

        else:
            y_pred=model.predict(X_test)
            if self.use_sigmoid==True:
                y_pred=1/(1+np.exp(-y_pred))

        #Model_ML.plot_predict_polarity(self, X_test,y_test,data_array,model)
        #Model_ML.plot_new_system(self, X_test,y_test,data_array,model)
        MSE,RMSE,MAE,R_square=Model_ML.plot_total_variance(self, y_test, y_pred)
        #plt.show()
        return y_pred,MSE,RMSE,MAE,R_square

class Conduct_Experiments(Model_ML):
    def __init__(self,config):
        super(Conduct_Experiments, self).__init__(config)

    def separation(self, X_test, y_test, y_pred, model):
        success_no_spapation=[]
        success_spapation = []
        failure_no_spapation=[]
        failure_spapation = []
        for i in range(X_test.shape[0]-1):
            for j in range(i,X_test.shape[0]):
                RE_i=np.abs(y_test[i]-y_pred[i])/y_test[i]
                RE_j = np.abs(y_test[j] - y_pred[j]) / y_test[j]
                if (X_test[i,167:173]==X_test[j,167:173]).all():
                    if y_test[i]<0.8 and y_test[i]>0.2:
                        if y_test[j]!=0:
                            if np.abs(y_test[i]-y_test[j])<=0.1 and np.abs(y_pred[i]-y_pred[j])<=0.1:
                                success_no_spapation.append(max(RE_i,RE_j))
                            elif np.abs(y_test[i]-y_test[j])>0.1 and np.abs(y_pred[i]-y_pred[j])>0.1:
                                success_spapation.append(max(RE_i,RE_j))
                            elif np.abs(y_test[i]-y_test[j])<=0.1 and np.abs(y_pred[i]-y_pred[j])>0.1:
                                failure_no_spapation.append(max(RE_i,RE_j))
                            elif np.abs(y_test[i]-y_test[j])>0.1 and np.abs(y_pred[i]-y_pred[j])<=0.1:
                                failure_spapation.append(max(RE_i,RE_j))
        success_spapation=np.array(success_spapation)
        success_no_spapation=np.array((success_no_spapation))
        failure_spapation=np.array(failure_spapation)
        failure_no_spapation=np.array(failure_no_spapation)
        print(f'successfully predict separation: {success_spapation.shape[0]}, mean relative error: {np.mean(success_spapation)}\n'
              f'successfully predict cannot separation: {success_no_spapation.shape[0]}, mean relative error: {np.mean(success_no_spapation)}\n'
              f'failure to predict separation: {failure_spapation.shape[0]}, mean relative error: {np.mean(failure_spapation)}\n'
              f'failure to predict cannot separation:{failure_no_spapation.shape[0]}, mean relative error: {np.mean(failure_no_spapation)}\n')
        print(min(failure_spapation),min(failure_no_spapation),np.percentile(failure_spapation,50),np.percentile(failure_no_spapation,50))

    def grid_search(self,X_train, y_train, X_validate, y_validate):
        '''
                train model using LightGBM,Xgboost,Random forest or ANN
                '''
        print('----------Start Training!--------------')

        torch.manual_seed(self.torch_seed)
        if self.use_model == 'LGB':
            parameters = {
                'max_depth': [1, 3, 5],
                'num_leaves': [5, 15, 25],
                'learning_rate': [0.0001, 0.0005, 0.0007]
            }
            if self.use_sigmoid == True:
                y_train = np.clip(y_train, 0.01, 0.99)
                y_validate = np.clip(y_validate, 0.01, 0.99)
                y_train = np.log(y_train / (1 - y_train))
                y_validate = np.log(y_validate / (1 - y_validate))
            model = lgb.LGBMRegressor(objective='regression', max_depth=self.LGB_max_depth,
                                      num_leaves=self.LGB_num_leaves,
                                      learning_rate=self.LGB_learning_rate, n_estimators=self.LGB_n_estimators)
            gsearch = GridSearchCV(model, param_grid=parameters, scoring='neg_mean_squared_error', cv=3)
            gsearch.fit(X_train, list(y_train.reshape(y_train.shape[0], )),
                      eval_set=[(X_train, list(y_train.reshape(y_train.shape[0], ))),
                                (X_validate, list(y_validate.reshape(y_validate.shape[0], )))],
                      eval_names=('fit', 'val'), eval_metric='l2', early_stopping_rounds=self.LGB_early_stopping_rounds,
                      verbose=False)
            print('参数的最佳取值:{0}'.format(gsearch.best_params_))
            print('最佳模型得分:{0}'.format(gsearch.best_score_))
            print(gsearch.cv_results_['mean_test_score'])
            print(gsearch.cv_results_['params'])
            print('----------LGB Training Finished!--------------')
            return model

        elif self.use_model == 'XGB':
            parameters = {
                'max_depth': [1,3,5],
                'n_estimators': [200, 300, 400],
                'learning_rate': [0.01, 0.1,1]
            }
            if self.use_sigmoid == True:
                y_train = np.clip(y_train, 0.01, 0.99)
                y_validate = np.clip(y_validate, 0.01, 0.99)
                y_train = np.log(y_train / (1 - y_train))
                y_validate = np.log(y_validate / (1 - y_validate))
            model = XGBRegressor(seed=self.seed,
                                 n_estimators=self.XGB_n_estimators,
                                 max_depth=self.XGB_max_depth,
                                 eval_metric='rmse',
                                 learning_rate=self.XGB_learning_rate,
                                 min_child_weight=1,
                                 subsample=1,
                                 colsample_bytree=1,
                                 colsample_bylevel=1,
                                 gamma=0)

            gsearch = GridSearchCV(model, param_grid=parameters, scoring='neg_mean_squared_error', cv=3)
            gsearch.fit(X_train, y_train.reshape(y_train.shape[0]))
            print('参数的最佳取值:{0}'.format(gsearch.best_params_))
            print('最佳模型得分:{0}'.format(gsearch.best_score_))
            print(gsearch.cv_results_['mean_test_score'])
            print(gsearch.cv_results_['params'])
            print('----------XGB Training Finished!--------------')
            return model

        elif self.use_model == 'RF':
            parameters = {
                'n_estimators': [1000, 2000, 3000],
            }
            if self.use_sigmoid == True:
                y_train = np.clip(y_train, 0.01, 0.99)
                y_validate = np.clip(y_validate, 0.01, 0.99)
                y_train = np.log(y_train / (1 - y_train))
                y_validate = np.log(y_validate / (1 - y_validate))
            model = RandomForestRegressor(n_estimators=self.RF_n_estimators,
                                          criterion='mse',
                                          random_state=self.RF_random_state,
                                          n_jobs=self.RF_n_jobs)
            gsearch = GridSearchCV(model, param_grid=parameters, scoring='neg_mean_squared_error', cv=3)
            gsearch.fit(X_train, y_train)
            print('参数的最佳取值:{0}'.format(gsearch.best_params_))
            print('最佳模型得分:{0}'.format(gsearch.best_score_))
            print(gsearch.cv_results_['mean_test_score'])
            print(gsearch.cv_results_['params'])
            print('----------RF Training Finished!--------------')
            return model


        elif self.use_model == 'Ensemble':
            y_train_origin = y_train.copy()
            y_validate_origin = y_validate.copy()
            if self.use_sigmoid == True:
                y_train = np.clip(y_train, 0.01, 0.99)
                y_validate = np.clip(y_validate, 0.01, 0.99)
                y_train = np.log(y_train / (1 - y_train))
                y_validate = np.log(y_validate / (1 - y_validate))
            model_LGB = lgb.LGBMRegressor(objective='regression', max_depth=self.LGB_max_depth,
                                          num_leaves=self.LGB_num_leaves,
                                          learning_rate=self.LGB_learning_rate, n_estimators=self.LGB_n_estimators)
            model_LGB.fit(X_train, list(y_train.reshape(y_train.shape[0], )),
                          eval_set=[(X_train, list(y_train.reshape(y_train.shape[0], ))),
                                    (X_validate, list(y_validate.reshape(y_validate.shape[0], )))],
                          eval_names=('fit', 'val'), eval_metric='l2',
                          early_stopping_rounds=self.LGB_early_stopping_rounds,
                          verbose=False)
            model_XGB = XGBRegressor(seed=self.seed,
                                     n_estimators=self.XGB_n_estimators,
                                     max_depth=self.XGB_max_depth,
                                     eval_metric='rmse',
                                     learning_rate=self.XGB_learning_rate,
                                     min_child_weight=1,
                                     subsample=1,
                                     colsample_bytree=1,
                                     colsample_bylevel=1,
                                     gamma=0)

            model_XGB.fit(X_train, y_train.reshape(y_train.shape[0]))
            model_RF = RandomForestRegressor(n_estimators=self.RF_n_estimators,
                                             criterion='mse',
                                             random_state=self.RF_random_state,
                                             n_jobs=self.RF_n_jobs)
            model_RF.fit(X_train, y_train)

            Net = ANN(X_train.shape[1], self.NN_hidden_neuron, 1, config=self.config).to(self.device)
            X_train = Variable(torch.from_numpy(X_train.astype(np.float32)).to(self.device), requires_grad=True)
            y_train = Variable(torch.from_numpy(y_train_origin.astype(np.float32)).to(self.device))
            X_validate = Variable(torch.from_numpy(X_validate.astype(np.float32)).to(self.device), requires_grad=True)
            y_validate = Variable(torch.from_numpy(y_validate_origin.astype(np.float32)).to(self.device))

            model_name = time.strftime('%m-%d-%H-%M', time.localtime(time.time()))
            dir_name = self.NN_model_save_location + '/' + model_name
            loss_plot = []
            loss_validate_plot = []
            try:
                os.makedirs(dir_name)
            except OSError:
                pass

            if self.NN_optimizer == 'SGD':
                optimizer = torch.optim.SGD(Net.parameters(), lr=self.NN_lr)
            elif self.NN_optimizer == 'RMSprop':
                optimizer = torch.optim.RMSprop(Net.parameters(), lr=self.NN_lr)
            else:
                optimizer = torch.optim.Adam(Net.parameters(), lr=self.NN_lr)

            with open(dir_name + '/' + 'data.txt', 'w') as f:  # 设置文件对象
                for epoch in range(self.NN_max_epoch):
                    optimizer.zero_grad()
                    prediction = Net(X_train)
                    prediction_validate = Net(X_validate)
                    dprediction = \
                        torch.autograd.grad(outputs=prediction[:, 0].sum(), inputs=X_train, create_graph=True)[0]
                    dprediction_deluent = dprediction[:, 167].reshape(X_train.shape[0], 1)
                    dprediction_validate = \
                        torch.autograd.grad(outputs=prediction_validate[:, 0].sum(), inputs=X_validate,
                                            create_graph=True)[0]
                    dprediction_validate_deluent = dprediction_validate[:, 167].reshape(X_validate.shape[0], 1)
                    MSELoss = torch.nn.MSELoss()
                    if self.NN_add_PINN == True:
                        loss = MSELoss(y_train, prediction) + self.NN_epi * (
                            torch.sum(F.relu(-dprediction_deluent))) / X_train.shape[0]
                        loss_validate = MSELoss(y_validate, prediction_validate) + (self.NN_epi * torch.sum(
                            F.relu(-dprediction_validate_deluent))) / X_validate.shape[0]
                    else:
                        loss = MSELoss(y_train, prediction)
                        loss_validate = MSELoss(y_validate, prediction_validate)

                    loss.backward()
                    optimizer.step()
                    if (epoch + 1) % 100 == 0:
                        print("iter_num: %d      loss: %.8f    loss_validate: %.8f" % (
                            epoch + 1, loss.item(), loss_validate.item()))
                        f.write("iter_num: %d      loss: %.8f    loss_validate: %.8f \r\n" % (
                            epoch + 1, loss.item(), loss_validate.item()))
                        torch.save(Net.state_dict(), dir_name + '/' + "%d_epoch.pkl" % (epoch + 1))
                        loss_plot.append(loss.item())
                        loss_validate_plot.append(loss_validate.item())

                best_epoch = (loss_validate_plot.index(min(loss_validate_plot)) + 1) * 100
                print("The ANN has been trained, the best epoch is %d" % (best_epoch))
                Net.load_state_dict(torch.load(dir_name + '/' + "%d_epoch.pkl" % (best_epoch)))
                Net.eval()


            model_ANN=Net
            X_valid_ANN = X_validate
            y_pred_ANN = model_ANN(X_valid_ANN).cpu().data.numpy()
            y_pred_ANN = y_pred_ANN.reshape(y_pred_ANN.shape[0], )
            X_validate=X_validate.cpu().data.numpy()
            y_validate = y_validate.cpu().data.numpy()

            y_pred_XGB = model_XGB.predict(X_validate)
            if self.use_sigmoid == True:
                y_pred_XGB = 1 / (1 + np.exp(-y_pred_XGB))

            y_pred_LGB = model_LGB.predict(X_validate)
            if self.use_sigmoid == True:
                y_pred_LGB = 1 / (1 + np.exp(-y_pred_LGB))

            y_pred_RF = model_RF.predict(X_validate)
            if self.use_sigmoid == True:
                y_pred_RF = 1 / (1 + np.exp(-y_pred_RF))

            weight=[0,0.1,0.2,0.3]
            for w in weight:
                y_pred = ( w* y_pred_LGB + w * y_pred_XGB + w * y_pred_RF + (1-3*w) * y_pred_ANN)

                MSE, RMSE, MAE, R_square = Model_ML.plot_total_variance(self, y_validate, y_pred)
                print(w,MSE, RMSE, MAE, R_square)

    def influence_of_weights(self,X_train, y_train, X_validate, y_validate,X_test,y_test):
        torch.manual_seed(self.torch_seed)
        y_train_origin = y_train.copy()
        y_validate_origin = y_validate.copy()
        if self.use_sigmoid == True:
            y_train = np.clip(y_train, 0.01, 0.99)
            y_validate = np.clip(y_validate, 0.01, 0.99)
            y_train = np.log(y_train / (1 - y_train))
            y_validate = np.log(y_validate / (1 - y_validate))
        model_LGB = lgb.LGBMRegressor(objective='regression', max_depth=self.LGB_max_depth,
                                      num_leaves=self.LGB_num_leaves,
                                      learning_rate=self.LGB_learning_rate, n_estimators=self.LGB_n_estimators)
        model_LGB.fit(X_train, list(y_train.reshape(y_train.shape[0], )),
                      eval_set=[(X_train, list(y_train.reshape(y_train.shape[0], ))),
                                (X_validate, list(y_validate.reshape(y_validate.shape[0], )))],
                      eval_names=('fit', 'val'), eval_metric='l2',
                      early_stopping_rounds=self.LGB_early_stopping_rounds,
                      verbose=False)
        model_XGB = XGBRegressor(seed=self.seed,
                                 n_estimators=self.XGB_n_estimators,
                                 max_depth=self.XGB_max_depth,
                                 eval_metric='rmse',
                                 learning_rate=self.XGB_learning_rate,
                                 min_child_weight=1,
                                 subsample=1,
                                 colsample_bytree=1,
                                 colsample_bylevel=1,
                                 gamma=0)

        model_XGB.fit(X_train, y_train.reshape(y_train.shape[0]))
        model_RF = RandomForestRegressor(n_estimators=self.RF_n_estimators,
                                         criterion='mse',
                                         random_state=self.RF_random_state,
                                         n_jobs=self.RF_n_jobs)
        model_RF.fit(X_train, y_train)

        Net = ANN(X_train.shape[1], self.NN_hidden_neuron, 1, config=self.config).to(self.device)
        X_train = Variable(torch.from_numpy(X_train.astype(np.float32)).to(self.device), requires_grad=True)
        y_train = Variable(torch.from_numpy(y_train_origin.astype(np.float32)).to(self.device))
        X_validate = Variable(torch.from_numpy(X_validate.astype(np.float32)).to(self.device), requires_grad=True)
        y_validate = Variable(torch.from_numpy(y_validate_origin.astype(np.float32)).to(self.device))

        model_name = time.strftime('%m-%d-%H-%M', time.localtime(time.time()))
        dir_name = self.NN_model_save_location + '/' + model_name
        loss_plot = []
        loss_validate_plot = []
        try:
            os.makedirs(dir_name)
        except OSError:
            pass

        if self.NN_optimizer == 'SGD':
            optimizer = torch.optim.SGD(Net.parameters(), lr=self.NN_lr)
        elif self.NN_optimizer == 'RMSprop':
            optimizer = torch.optim.RMSprop(Net.parameters(), lr=self.NN_lr)
        else:
            optimizer = torch.optim.Adam(Net.parameters(), lr=self.NN_lr)

        with open(dir_name + '/' + 'data.txt', 'w') as f:  # 设置文件对象
            for epoch in range(self.NN_max_epoch):
                optimizer.zero_grad()
                prediction = Net(X_train)
                prediction_validate = Net(X_validate)
                dprediction = \
                    torch.autograd.grad(outputs=prediction[:, 0].sum(), inputs=X_train, create_graph=True)[0]
                dprediction_deluent = dprediction[:, 167].reshape(X_train.shape[0], 1)
                dprediction_validate = \
                    torch.autograd.grad(outputs=prediction_validate[:, 0].sum(), inputs=X_validate,
                                        create_graph=True)[0]
                dprediction_validate_deluent = dprediction_validate[:, 167].reshape(X_validate.shape[0], 1)
                MSELoss = torch.nn.MSELoss()
                if self.NN_add_PINN == True:
                    loss = MSELoss(y_train, prediction) + self.NN_epi * (
                        torch.sum(F.relu(-dprediction_deluent))) / X_train.shape[0]
                    loss_validate = MSELoss(y_validate, prediction_validate) + (self.NN_epi * torch.sum(
                        F.relu(-dprediction_validate_deluent))) / X_validate.shape[0]
                else:
                    loss = MSELoss(y_train, prediction)
                    loss_validate = MSELoss(y_validate, prediction_validate)

                loss.backward()
                optimizer.step()
                if (epoch + 1) % 100 == 0:
                    print("iter_num: %d      loss: %.8f    loss_validate: %.8f" % (
                        epoch + 1, loss.item(), loss_validate.item()))
                    f.write("iter_num: %d      loss: %.8f    loss_validate: %.8f \r\n" % (
                        epoch + 1, loss.item(), loss_validate.item()))
                    torch.save(Net.state_dict(), dir_name + '/' + "%d_epoch.pkl" % (epoch + 1))
                    loss_plot.append(loss.item())
                    loss_validate_plot.append(loss_validate.item())

            best_epoch = (loss_validate_plot.index(min(loss_validate_plot)) + 1) * 100
            print("The ANN has been trained, the best epoch is %d" % (best_epoch))
            Net.load_state_dict(torch.load(dir_name + '/' + "%d_epoch.pkl" % (best_epoch)))
            Net.eval()

        model_ANN = Net
        X_test_ANN = Variable(torch.from_numpy(X_test.astype(np.float32)).to(self.device), requires_grad=True)
        y_pred_ANN = model_ANN(X_test_ANN).cpu().data.numpy()
        y_pred_ANN = y_pred_ANN.reshape(y_pred_ANN.shape[0], )

        y_pred_XGB = model_XGB.predict(X_test)
        if self.use_sigmoid == True:
            y_pred_XGB = 1 / (1 + np.exp(-y_pred_XGB))

        y_pred_LGB = model_LGB.predict(X_test)
        if self.use_sigmoid == True:
            y_pred_LGB = 1 / (1 + np.exp(-y_pred_LGB))

        y_pred_RF = model_RF.predict(X_test)
        if self.use_sigmoid == True:
            y_pred_RF = 1 / (1 + np.exp(-y_pred_RF))


        weight = [0,0.05, 0.1,0.15, 0.2,0.25, 0.3]
        for w in weight:
            y_pred = (w * y_pred_LGB + w * y_pred_XGB + w * y_pred_RF + (1 - 3 * w) * y_pred_ANN)

            MSE, RMSE, MAE, R_square = Model_ML.plot_total_variance(self, y_test, y_pred)
            print(w, MSE, RMSE, MAE, R_square)


def generate_cartesian_corrdinates(SMILES):
    Mol = SMILES.copy()  # Stores mol information of a compound
    for i in range(len(Mol)):
        Mol[i] = Chem.MolFromSmiles(SMILES[i])

    Three_D = []  # Store 3D conformation information

    for i in Mol:
        m1 = Chem.AddHs(i)  # Add hydrogen atoms
        AllChem.EmbedMolecule(m1)  # 3D conformation generation using ETKDG optimization
        Three_D.append(m1)

    # Generate Cartesian coordinates and write to txt file
    f1 = open('coordinate.txt', 'a')
    for i in range(len(Three_D)):
        coor = Chem.MolToMolBlock(Three_D[i])  # Generate Cartesian coordinates
        f1.write('\n')
        f1.write(coor)
    f1.close()

def main():
    config = parse_args()
    config.seed=324
    Data = Dataset_process(config)
    X_train, y_train, X_validate, y_validate, X_test, y_test, data_array = Data.split_dataset()
    Model = Model_ML(config)
    model = Model.train(X_train, y_train, X_validate, y_validate)
    y_pred, MSE, RMSE, MAE, R_square = Model.test(X_test, y_test, data_array, model)
    print(MSE, RMSE, MAE,R_square)

def grid_search():
    config = parse_args()
    Data = Dataset_process(config)
    X_train, y_train, X_validate, y_validate, X_test, y_test, data_array = Data.split_dataset()
    Model = Model_ML(config)
    Experiment = Conduct_Experiments(config)
    #Experiment.grid_search(X_train, y_train, X_validate, y_validate)
    Experiment.influence_of_weights(X_train, y_train, X_validate, y_validate, X_test, y_test)

def compare_different_method():
    model_select = ['XGB', 'LGB', 'ANN', 'Ensemble', 'Bayesian', 'RF']
    for i in range(6):
        config = parse_args()
        config.use_model = model_select[i]
        config.seed=732
        Data = Dataset_process(config)
        X, y, ID = Data.split_dataset_all()
        print(X.shape)
        np.random.seed(config.seed)
        state = np.random.get_state()
        np.random.shuffle(X)
        np.random.set_state(state)
        np.random.shuffle(y)
        total = X.shape[0]
        X_train = X[0:int(total * 0.8)]
        y_train = y[0:int(total * 0.8)]
        X_validate = X[int(total * 0.8):int(total * 0.9)]
        y_validate = y[int(total * 0.8):int(total * 0.9)]
        X_test = X[int(total * 0.9):total]
        y_test = y[int(total * 0.9):total]
        Model = Model_ML(config)
        model = Model.train(X_train, y_train, X_validate, y_validate)
        y_pred, MSE, RMSE, MAE, R_square = Model.test(X_test, y_test, [], model)
        print(model_select[i],MSE, RMSE, MAE, R_square)

def multiple_trails_all():
    config = parse_args()
    record = np.zeros([6, 4, 10])
    seed = [732, 256, 23, 1234, 425, 644, 888, 912, 1198, 1032]
    model_select = ['XGB', 'LGB', 'ANN', 'Ensemble', 'Bayesian', 'RF']
    for k in range(6):
        config.use_model = model_select[k]
        for i in range(10):
            config.seed=seed[i]
            Data = Dataset_process(config)
            X, y,ID = Data.split_dataset_all()
            np.random.seed(config.seed)
            state = np.random.get_state()
            np.random.shuffle(X)
            np.random.set_state(state)
            np.random.shuffle(y)
            total = X.shape[0]
            X_train = X[0:int(total * 0.8)]
            y_train = y[0:int(total * 0.8)]
            X_validate = X[int(total * 0.8):int(total * 0.9)]
            y_validate = y[int(total * 0.8):int(total * 0.9)]
            X_test = X[int(total * 0.9):total]
            y_test = y[int(total * 0.9):total]
            Model = Model_ML(config)
            model = Model.train(X_train, y_train, X_validate, y_validate)
            y_pred, MSE, RMSE, MAE, R_square = Model.test(X_test, y_test, [], model)
            record[k, 0, i] = MSE
            record[k, 1, i] = RMSE
            record[k, 2, i] = MAE
            record[k, 3, i] = R_square
            print(record[k, :, i])
    print(record)
    np.save("result_save/revise_multiple_trail.npy", record)

def different_num():
    config = parse_args()
    training_num = [0.025, 0.05, 0.1, 0.2,0.3,0.4, 0.5, 0.6,0.7, 0.8]
    MSE_record=[]
    MAE_record=[]
    R_square_record=[]
    for i in training_num:
        config.choose_train = int(i*config.choose_total)
        config.test_mode = 'fix'
        Data = Dataset_process(config)
        X_train, y_train, X_validate, y_validate, X_test, y_test, data_array = Data.split_dataset()
        Model = Model_ML(config)
        model = Model.train(X_train, y_train, X_validate, y_validate)
        y_pred, MSE, RMSE, MAE, R_square = Model.test(X_test, y_test, data_array, model)
        MSE_record.append(MSE)
        MAE_record.append(MAE)
        R_square_record.append(R_square)
    print(MSE_record)
    print(MAE_record)
    print(R_square_record)
    np.save('result_save/revised_MSE_different_num',MSE_record)
    np.save('result_save/revised_MAE_different_num', MAE_record)
    np.save('result_save/revised_R_square_different_num', R_square_record)

def get_importance_shuffle():
    config = parse_args()
    Data = Dataset_process(config)
    X_train, y_train, X_validate, y_validate, X_test, y_test, data_array = Data.split_dataset()
    Model = Model_ML(config)
    record = np.zeros([4, 8])
    model = Model.train(X_train, y_train, X_validate, y_validate)
    y_pred, MSE, RMSE, MAE, R_square = Model.test(X_test, y_test, data_array, model)
    record[0, 0] = MSE
    record[1, 0] = RMSE
    record[2, 0] = MAE
    record[3, 0] = R_square
    for i in range(173, 180):
        X_test_change = X_test.copy()
        X_test_change[:,i]=np.random.uniform(0,1,size=X_test_change[:,i].shape)*(np.max(X_test_change[:,i])-np.min(X_test_change[:,i]))+np.min(X_test_change[:,i])
        y_pred, MSE, RMSE, MAE, R_square = Model.test(X_test_change, y_test, data_array, model)
        record[0, i - 172] = MSE
        record[1, i - 172] = RMSE
        record[2, i - 172] = MAE
        record[3, i - 172] = R_square
    print(record)
    np.save("result_save/revised_importance_Ensemble_shuffle", record)

def predict_single(smile,dipole=-1):
    config = parse_args()
    if dipole==-1:
        config.add_dipole = False
        Data = Dataset_process(config)
        X_train, y_train, X_validate, y_validate, X_test, y_test, data_array = Data.split_dataset()
        Model = Model_ML(config)
        model = Model.train(X_train, y_train, X_validate, y_validate)
        y_pred, MSE, RMSE, MAE, R_square = Model.test(X_test, y_test, data_array, model)

        compound_mol = Chem.MolFromSmiles(smile)
        Finger = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smile))
        fingerprint = np.array([x for x in Finger])
        compound_finger = fingerprint
        compound_MolWt = Descriptors.ExactMolWt(compound_mol)
        compound_TPSA = Chem.rdMolDescriptors.CalcTPSA(compound_mol)
        print(compound_TPSA)
        compound_nRotB = Descriptors.NumRotatableBonds(compound_mol)  # Number of rotable bonds
        compound_HBD = Descriptors.NumHDonors(compound_mol)  # Number of H bond donors
        compound_HBA = Descriptors.NumHAcceptors(compound_mol)  # Number of H bond acceptors
        compound_LogP = Descriptors.MolLogP(compound_mol)  # LogP
        X_test = np.zeros([1, 179])
        X_test[0, 0:167] = compound_finger
        X_test[0, 167:173] = 0
        X_test[0, 173:179] = [compound_MolWt, compound_TPSA, compound_nRotB, compound_HBD, compound_HBA, compound_LogP]
    else:
        config.add_dipole = True
        Data = Dataset_process(config)
        X_train, y_train, X_validate, y_validate, X_test, y_test, data_array = Data.split_dataset()
        Model = Model_ML(config)
        model = Model.train(X_train, y_train, X_validate, y_validate)
        y_pred, MSE, RMSE, MAE, R_square = Model.test(X_test, y_test, data_array, model)
        print(R_square)

        compound_mol = Chem.MolFromSmiles(smile)
        Finger = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smile))
        fingerprint = np.array([x for x in Finger])
        compound_finger = fingerprint
        compound_MolWt = Descriptors.ExactMolWt(compound_mol)
        compound_TPSA = Chem.rdMolDescriptors.CalcTPSA(compound_mol)
        print(compound_TPSA)
        compound_nRotB = Descriptors.NumRotatableBonds(compound_mol)  # Number of rotable bonds
        compound_HBD = Descriptors.NumHDonors(compound_mol)  # Number of H bond donors
        compound_HBA = Descriptors.NumHAcceptors(compound_mol)  # Number of H bond acceptors
        compound_LogP = Descriptors.MolLogP(compound_mol)  # LogP
        X_test = np.zeros([1, 180])
        X_test[0, 0:167] = compound_finger
        X_test[0, 167:173] = 0
        X_test[0,174]=dipole
        X_test[0, 174:180] = [compound_MolWt, compound_TPSA, compound_nRotB, compound_HBD, compound_HBA, compound_LogP]
    y=np.array([0.23]).reshape([1,1])
    X_test = X_test.copy()
    eluent_origin = np.array([[1, 0, 0, 0, 0], [0.980392, 0.019608, 0, 0, 0], [0.952381, 0.047619, 0, 0, 0],
                  [0.833333, 0.166667, 0, 0, 0], [0.75, 0.25, 0, 0, 0], [0.5, 0.5, 0, 0, 0],
                  [0.333333, 0.666667, 0, 0, 0], [0, 1, 0, 0, 0],[0, 0, 1, 0, 0], [0, 0, 0.990099, 0.009901, 0], [0, 0, 0.980392, 0.019608, 0],
                        [0, 0, 0.967742, 0.032258, 0], [0, 0, 0.952381, 0.047619, 0], [0, 0, 0.909091, 0.090909, 0]], dtype=np.float32)
    eluent = []

    for i in range(eluent_origin.shape[0]):
        eluent.append(Data.get_eluent_descriptor(eluent_origin[i]))
    eluent = np.array(eluent)
    print(smile)
    for i in range(eluent.shape[0]):
        X_test[0,167:173] = eluent[i]
        y_pred, MSE, RMSE, MAE, R_square = Model.test(X_test.reshape(1, X_test.shape[1]),
                                                        y[0],
                                                        data_array, model)

        print(y_pred[0])

def predict_sugar():
    smiles=['O=C(OC1C(OC(C)=O)C(OC(C)=O)C(OC(C)=O)C(COC(C)=O)O1)C',
'CC(OC[C@H]1O[C@@H](OC(C)=O)[C@H](OC(C)=O)[C@@H](OC(C)=O)[C@H]1OC(C)=O)=O',
'OC1O[C@H](COCC2=CC=CC=C2)[C@@H](OCC3=CC=CC=C3)[C@H](OCC4=CC=CC=C4)[C@H]1OCC5=CC=CC=C5',
'CC(OC[C@@H]1[C@@H](OC(C)=O)[C@H](OC(C)=O)C=CO1)=O']
    diploe=[4.707,6.66,2.756,3.295]
    for i in range(4):
        predict_single(smiles[i],diploe[i])

def compare_constraints():
    config = parse_args()
    record = np.zeros([2, 4, 10])
    seed =np.random.randint(0,10000,10)
    config.use_model = 'Ensemble'
    for k in range(2):
        if k==0:
            config.use_sigmoid=False
        if k==1:
            config.use_sigmoid = True
        for i in range(10):
            config.seed = seed[i]
            Data = Dataset_process(config)
            X_train, y_train, X_validate, y_validate, X_test, y_test, data_array = Data.split_dataset()
            print(X_train.shape, X_validate.shape, X_test.shape)
            Model = Model_ML(config)
            model = Model.train(X_train, y_train, X_validate, y_validate)
            y_pred, MSE, RMSE, MAE, R_square = Model.test(X_test, y_test, data_array, model)
            record[k, 0, i] = MSE
            record[k, 1, i] = RMSE
            record[k, 2, i] = MAE
            record[k, 3, i] = R_square
            print(record[k, :, i])
    print(record)
    np.save("result_save/revised_record_sigmoid.npy", record)
if __name__ == '__main__':
    #compare_constraints()
    #get_importance_shuffle()
    #compare_different_method()
    #compare_precise_TPSA()
    main()
    #grid_search()
    #multiple_trails_all()
    #different_num()
