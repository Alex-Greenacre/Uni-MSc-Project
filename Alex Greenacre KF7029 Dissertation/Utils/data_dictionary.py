#####
#File - Data Dictionary 
#Description - Stored information of each scenario file and functions used to get the file name for file loading 
####
files=[
    ['m1_radar.mat', 'm1_reference.mat', 1,  'F' ],
    ['m2_radar.mat', 'm2_reference.mat', 1,  'F' ],
    ['m3_radar.mat', 'm3_reference.mat', 1,  'F' ],
    ['m4_radar.mat', 'm4_reference.mat', 1,  'D' ],
    ['m5_radar.mat', 'm5_reference.mat', 1,  'F' ],
    ['m6_radar.mat', 'm6_reference.mat', 1,  'F' ],
    ['m7_radar.mat', 'm7_reference.mat', 1,  'F' ],
    ['m8_radar.mat', 'm8_reference.mat', 1,  'D' ],
    ['m9_radar.mat', 'm9_reference.mat', 1,  'W' ],
    ['m10_radar.mat', 'm10_reference.mat', 1,  'W' ],
    ['m11_radar.mat', 'm11_reference.mat', 1,  'W' ],
    ['m12_radar.mat', 'm12_reference.mat', 5,  'F' ],
    ['m13_radar.mat', 'm13_reference.mat', 5,  'F' ],
    ['m14_radar.mat', 'm14_reference.mat', 5,  'F' ],
    ['m15_radar.mat', 'm15_reference.mat', 5,  'F' ],
    ['m16_radar.mat', 'm16_reference.mat', 5,  'F' ],
    ['m17_radar.mat', 'm17_reference.mat', 5,  'F' ],
    ['m18_radar.mat', 'm18_reference.mat', 5,  'F' ],
    ['m19_radar.mat', 'm19_reference.mat', 5,  'F' ],
    ['m20_radar.mat', 'm20_reference.mat', 5,  'F' ],
    ['m21_radar.mat', 'm21_reference.mat', 5,  'F' ],
    ['m22_radar.mat', 'm22_reference.mat', 5,  'F' ],
    ['m23_radar.mat', 'm23_reference.mat', 5,  'F' ],
    ['m24_radar.mat', 'm24_reference.mat', 5,  'F' ],
    ['m25_radar.mat', 'm25_reference.mat', 4,  'F' ],
    ['m26_radar.mat', 'm26_reference.mat', 4,  'F' ],
    ['m27_radar.mat', 'm27_reference.mat', 3,  'F' ],
    ['m27-1_radar.mat', 'm27-1_reference.mat', 3,  'W' ],
    ['m27-2_radar.mat', 'm27-2_reference.mat', 3,  'D' ], 
    ['m28_radar.mat', 'm28_reference.mat', 3,  'F' ],
    ['m28-1_radar.mat', 'm28-1_reference.mat', 3,  'W' ],
    ['m28-2_radar.mat', 'm28-2_reference.mat', 3,  'D' ], 
    ['m29_radar.mat', 'm29_reference.mat', 3,  'F' ],
    ['m29-1_radar.mat', 'm29-1_reference.mat', 3,  'W' ],
    ['m29-2_radar.mat', 'm29-2_reference.mat', 3,  'D' ], 
    ['m30_radar.mat', 'm30_reference.mat', 3,  'F' ],
    ['m30-1_radar.mat', 'm30-1_reference.mat', 3,  'W' ],
    ['m30-2_radar.mat', 'm30-2_reference.mat', 3,  'D' ], 
    ['m31_radar.mat', 'm31_reference.mat', 3,  'F' ],
    ['m31-1_radar.mat', 'm31-1_reference.mat', 3,  'W' ],
    ['m31-2_radar.mat', 'm31-2_reference.mat', 3,  'D' ], 
    ['m32_radar.mat', 'm32_reference.mat', 3,  'F' ],
    ['m32-1_radar.mat', 'm32-1_reference.mat', 3,  'W' ],
    ['m32-2_radar.mat', 'm32-2_reference.mat', 3,  'D' ], 
    ['m33_radar.mat', 'm33_reference.mat', 2,  'F' ],
    ['m33-1_radar.mat', 'm33-1_reference.mat', 2,  'W' ],
    ['m33-2_radar.mat', 'm33-2_reference.mat', 2,  'D' ], 
    ['m34_radar.mat', 'm34_reference.mat', 2,  'F' ],
    ['m34-1_radar.mat', 'm34-1_reference.mat', 2,  'W' ],
    ['m34-2_radar.mat', 'm34-2_reference.mat', 2,  'D' ], 
    ['m35_radar.mat', 'm35_reference.mat', 2,  'F' ],
    ['m35-1_radar.mat', 'm35-1_reference.mat', 2,  'W' ],
    ['m35-2_radar.mat', 'm35-2_reference.mat', 2,  'D' ], 
    ['m36_radar.mat', 'm36_reference.mat', 2,  'F' ],
    ['m36-1_radar.mat', 'm36-1_reference.mat', 2,  'W' ],
    ['m36-2_radar.mat', 'm36-2_reference.mat', 2,  'D' ], 
    ['m37_radar.mat', 'm37_reference.mat', 2,  'F' ],
    ['m37-1_radar.mat', 'm37-1_reference.mat', 2,  'W' ],
    ['m37-2_radar.mat', 'm37-2_reference.mat', 2,  'D' ]
]
def getDataFiles(pNumberOfPeople):
    data_files =[]
    reference_files = []
    for i in files:
        if i[2] == pNumberOfPeople:
            data_files.append(i[0])
            reference_files.append(i[1])
    return data_files,reference_files
#Due to data distribution this fucntion will be locked to 2,3 person scenarios to garentee an even distribution of data    
def getDataFilesWithObject(pObject):
    data_files =[]
    reference_files = []
    for i in files:
        if (i[2] == 2 or i[2]==3) and i[3]==pObject:
            data_files.append(i[0])
            reference_files.append(i[1])
    return data_files,reference_files
def getAllFiles():
    data_files =[]
    reference_files = []
    for i in files:
        data_files.append(i[0])
        reference_files.append(i[1])
    return data_files,reference_files
