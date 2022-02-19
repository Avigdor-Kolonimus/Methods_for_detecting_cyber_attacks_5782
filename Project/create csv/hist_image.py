import os
import numpy as np
import cv2 as cv
from sklearn.cluster import MiniBatchKMeans
import os
import pandas as pd # CSV handling with operations on tabular data.

column_names = [
        'filename', 
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', 
        '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', 
        '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', 
        '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', 
        '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', 
        '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', 
        '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', 
        '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', 
        '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', 
        '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196', '197', '198', '199', 
        '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210', '211', '212', '213', '214', '215', '216', '217', '218', '219', 
        '220', '221', '222', '223', '224', '225', '226', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '237', '238', '239', 
        '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251', '252', '253', '254', '255'
        , 'label'
        ]
my_e_df = pd.DataFrame(columns = column_names)
my_q_df = pd.DataFrame(columns = column_names)
my_g_df = pd.DataFrame(columns = column_names)

if __name__ == "__main__":
    # Read Data
    directorpaths =[f'./Mal/', f'./Ben_1/', f'./Ben_2/']
    # directorpaths =[f'./Mal_test/', f'./Ben_test/']
  
    # Read Markers
    for directorypath in directorpaths:
        label_dir = 0
        dir_list = os.listdir(directorypath)
        if directorypath.find('Mal') != -1:
            label_dir = 1
        for imagename in dir_list:
            imagePath = f'{directorypath}{imagename}'
            
            try:
                # load the image and grab its width and height
                img = cv.imread(imagePath)
                (h, w) = img.shape[:2]
                cntPixel = h * w

                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                gray_histr = cv.calcHist([gray],[0],None,[256],[0,256])
                gray_histr = gray_histr / cntPixel
            
                equalize = cv.equalizeHist(gray)
                equalize_histr = cv.calcHist([equalize],[0],None,[256],[0,256])
                equalize_histr = equalize_histr / cntPixel

                lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
                lab = lab.reshape((lab.shape[0] * lab.shape[1], 3))

                # apply k-means using the specified number of clusters and
                # then create the quantized image based on the predictions
                clt = MiniBatchKMeans(n_clusters = 16)
                labels = clt.fit_predict(lab)
                quant = clt.cluster_centers_.astype("uint8")[labels]

                # reshape the feature vectors to images
                quant = quant.reshape((h, w, 3))

                # convert from L*a*b* to RGB
                tmp_quant = cv.cvtColor(quant, cv.COLOR_LAB2BGR)
                quant = cv.cvtColor(tmp_quant, cv.COLOR_BGR2GRAY)
                quant_histr = cv.calcHist([quant],[0],None,[256],[0,256])
                quant_histr = quant_histr / cntPixel

                new_e_row = {'filename': imagename,
                '0' : equalize_histr[0][0], '1' : equalize_histr[1][0], '2' : equalize_histr[2][0], '3' : equalize_histr[3][0], '4' : equalize_histr[4][0], 
                '5' : equalize_histr[5][0], '6' : equalize_histr[6][0], '7' : equalize_histr[7][0], '8' : equalize_histr[8][0], '9' : equalize_histr[9][0], 
                '10' : equalize_histr[10][0], '11' : equalize_histr[11][0], '12' : equalize_histr[12][0], '13' : equalize_histr[13][0], '14' : equalize_histr[14][0], 
                '15' : equalize_histr[15][0], '16' : equalize_histr[16][0], '17' : equalize_histr[17][0], '18' : equalize_histr[18][0], '19' : equalize_histr[20][0],
                '20' : equalize_histr[20][0], '21' : equalize_histr[21][0], '22' : equalize_histr[22][0], '23' : equalize_histr[23][0], '24' : equalize_histr[24][0], 
                '25' : equalize_histr[25][0], '26' : equalize_histr[26][0], '27' : equalize_histr[27][0], '28' : equalize_histr[28][0], '29' : equalize_histr[29][0],
                '30' : equalize_histr[30][0], '31' : equalize_histr[31][0], '32' : equalize_histr[32][0], '33' : equalize_histr[33][0], '34' : equalize_histr[34][0], 
                '35' : equalize_histr[35][0], '36' : equalize_histr[36][0], '37' : equalize_histr[37][0], '38' : equalize_histr[38][0], '39' : equalize_histr[39][0],
                '40' : equalize_histr[40][0], '41' : equalize_histr[41][0], '42' : equalize_histr[42][0], '43' : equalize_histr[43][0], '44' : equalize_histr[44][0], 
                '45' : equalize_histr[45][0], '46' : equalize_histr[46][0], '47' : equalize_histr[47][0], '48' : equalize_histr[48][0], '49' : equalize_histr[49][0], 
                '50' : equalize_histr[50][0], '51' : equalize_histr[51][0], '52' : equalize_histr[52][0], '53' : equalize_histr[53][0], '54' : equalize_histr[54][0], 
                '55' : equalize_histr[55][0], '56' : equalize_histr[56][0], '57' : equalize_histr[57][0], '58' : equalize_histr[58][0], '59' : equalize_histr[59][0], 
                '60' : equalize_histr[60][0], '61' : equalize_histr[61][0], '62' : equalize_histr[62][0], '63' : equalize_histr[63][0], '64' : equalize_histr[64][0], 
                '65' : equalize_histr[65][0], '66' : equalize_histr[66][0], '67' : equalize_histr[67][0], '68' : equalize_histr[68][0], '69' : equalize_histr[69][0], 
                '70' : equalize_histr[70][0], '71' : equalize_histr[71][0], '72' : equalize_histr[72][0], '73' : equalize_histr[73][0], '74' : equalize_histr[74][0], 
                '75' : equalize_histr[75][0], '76' : equalize_histr[76][0], '77' : equalize_histr[77][0], '78' : equalize_histr[78][0], '79' : equalize_histr[79][0], 
                '80' : equalize_histr[80][0], '81' : equalize_histr[81][0], '82' : equalize_histr[82][0], '83' : equalize_histr[83][0], '84' : equalize_histr[84][0], 
                '85' : equalize_histr[85][0], '86' : equalize_histr[86][0], '87' : equalize_histr[87][0], '88' : equalize_histr[88][0], '89' : equalize_histr[89][0], 
                '90' : equalize_histr[90][0], '91' : equalize_histr[91][0], '92' : equalize_histr[92][0], '93' : equalize_histr[93][0], '94' : equalize_histr[94][0], 
                '95' : equalize_histr[95][0], '96' : equalize_histr[96][0], '97' : equalize_histr[97][0], '98' : equalize_histr[98][0], '99' : equalize_histr[99][0], 
                '100' : equalize_histr[100][0], '101' : equalize_histr[101][0], '102' : equalize_histr[102][0], '103' : equalize_histr[103][0], '104' : equalize_histr[104][0], 
                '105' : equalize_histr[105][0], '106' : equalize_histr[106][0], '107' : equalize_histr[107][0], '108' : equalize_histr[108][0], '109' : equalize_histr[109][0], 
                '110' : equalize_histr[110][0], '111' : equalize_histr[111][0], '112' : equalize_histr[112][0], '113' : equalize_histr[113][0], '114' : equalize_histr[114][0], 
                '115' : equalize_histr[115][0], '116' : equalize_histr[116][0], '117' : equalize_histr[117][0], '118' : equalize_histr[118][0], '119' : equalize_histr[120][0],
                '120' : equalize_histr[120][0], '121' : equalize_histr[121][0], '122' : equalize_histr[122][0], '123' : equalize_histr[123][0], '124' : equalize_histr[124][0], 
                '125' : equalize_histr[125][0], '126' : equalize_histr[126][0], '127' : equalize_histr[127][0], '128' : equalize_histr[128][0], '129' : equalize_histr[129][0],
                '130' : equalize_histr[130][0], '131' : equalize_histr[131][0], '132' : equalize_histr[132][0], '133' : equalize_histr[133][0], '134' : equalize_histr[134][0], 
                '135' : equalize_histr[135][0], '136' : equalize_histr[136][0], '137' : equalize_histr[137][0], '138' : equalize_histr[138][0], '139' : equalize_histr[139][0],
                '140' : equalize_histr[140][0], '141' : equalize_histr[141][0], '142' : equalize_histr[142][0], '143' : equalize_histr[143][0], '144' : equalize_histr[144][0], 
                '145' : equalize_histr[145][0], '146' : equalize_histr[146][0], '147' : equalize_histr[147][0], '148' : equalize_histr[148][0], '149' : equalize_histr[149][0], 
                '150' : equalize_histr[150][0], '151' : equalize_histr[151][0], '152' : equalize_histr[152][0], '153' : equalize_histr[153][0], '154' : equalize_histr[154][0], 
                '155' : equalize_histr[155][0], '156' : equalize_histr[156][0], '157' : equalize_histr[157][0], '158' : equalize_histr[158][0], '159' : equalize_histr[159][0], 
                '160' : equalize_histr[160][0], '161' : equalize_histr[161][0], '162' : equalize_histr[162][0], '163' : equalize_histr[163][0], '164' : equalize_histr[164][0], 
                '165' : equalize_histr[165][0], '166' : equalize_histr[166][0], '167' : equalize_histr[167][0], '168' : equalize_histr[168][0], '169' : equalize_histr[169][0], 
                '170' : equalize_histr[170][0], '171' : equalize_histr[171][0], '172' : equalize_histr[172][0], '173' : equalize_histr[173][0], '174' : equalize_histr[174][0], 
                '175' : equalize_histr[175][0], '176' : equalize_histr[176][0], '177' : equalize_histr[177][0], '178' : equalize_histr[178][0], '179' : equalize_histr[179][0], 
                '180' : equalize_histr[180][0], '181' : equalize_histr[181][0], '182' : equalize_histr[182][0], '183' : equalize_histr[183][0], '184' : equalize_histr[184][0], 
                '185' : equalize_histr[185][0], '186' : equalize_histr[186][0], '187' : equalize_histr[187][0], '188' : equalize_histr[188][0], '189' : equalize_histr[189][0], 
                '190' : equalize_histr[190][0], '191' : equalize_histr[191][0], '192' : equalize_histr[192][0], '193' : equalize_histr[193][0], '194' : equalize_histr[194][0], 
                '195' : equalize_histr[195][0], '196' : equalize_histr[196][0], '197' : equalize_histr[197][0], '198' : equalize_histr[198][0], '199' : equalize_histr[199][0],
                '200' : equalize_histr[200][0], '201' : equalize_histr[201][0], '202' : equalize_histr[202][0], '203' : equalize_histr[203][0], '204' : equalize_histr[204][0], 
                '205' : equalize_histr[205][0], '206' : equalize_histr[206][0], '207' : equalize_histr[207][0], '208' : equalize_histr[208][0], '209' : equalize_histr[209][0], 
                '210' : equalize_histr[210][0], '211' : equalize_histr[211][0], '212' : equalize_histr[212][0], '213' : equalize_histr[213][0], '214' : equalize_histr[214][0], 
                '215' : equalize_histr[215][0], '216' : equalize_histr[216][0], '217' : equalize_histr[217][0], '218' : equalize_histr[218][0], '219' : equalize_histr[220][0],
                '220' : equalize_histr[220][0], '221' : equalize_histr[221][0], '222' : equalize_histr[222][0], '223' : equalize_histr[223][0], '224' : equalize_histr[224][0], 
                '225' : equalize_histr[225][0], '226' : equalize_histr[226][0], '227' : equalize_histr[227][0], '228' : equalize_histr[228][0], '229' : equalize_histr[229][0],
                '230' : equalize_histr[230][0], '231' : equalize_histr[231][0], '232' : equalize_histr[232][0], '233' : equalize_histr[233][0], '234' : equalize_histr[234][0], 
                '235' : equalize_histr[235][0], '236' : equalize_histr[236][0], '237' : equalize_histr[237][0], '238' : equalize_histr[238][0], '239' : equalize_histr[239][0],
                '240' : equalize_histr[240][0], '241' : equalize_histr[241][0], '242' : equalize_histr[242][0], '243' : equalize_histr[243][0], '244' : equalize_histr[244][0], 
                '245' : equalize_histr[245][0], '246' : equalize_histr[246][0], '247' : equalize_histr[247][0], '248' : equalize_histr[248][0], '249' : equalize_histr[249][0], 
                '250' : equalize_histr[250][0], '251' : equalize_histr[251][0], '252' : equalize_histr[252][0], '253' : equalize_histr[253][0], '254' : equalize_histr[254][0], 
                '255' : equalize_histr[255][0],
                'label': label_dir}
                
                new_q_row = {'filename': imagename,
                '0' : quant_histr[0][0], '1' : quant_histr[1][0], '2' : quant_histr[2][0], '3' : quant_histr[3][0], '4' : quant_histr[4][0], 
                '5' : quant_histr[5][0], '6' : quant_histr[6][0], '7' : quant_histr[7][0], '8' : quant_histr[8][0], '9' : quant_histr[9][0], 
                '10' : quant_histr[10][0], '11' : quant_histr[11][0], '12' : quant_histr[12][0], '13' : quant_histr[13][0], '14' : quant_histr[14][0], 
                '15' : quant_histr[15][0], '16' : quant_histr[16][0], '17' : quant_histr[17][0], '18' : quant_histr[18][0], '19' : quant_histr[20][0],
                '20' : quant_histr[20][0], '21' : quant_histr[21][0], '22' : quant_histr[22][0], '23' : quant_histr[23][0], '24' : quant_histr[24][0], 
                '25' : quant_histr[25][0], '26' : quant_histr[26][0], '27' : quant_histr[27][0], '28' : quant_histr[28][0], '29' : quant_histr[29][0],
                '30' : quant_histr[30][0], '31' : quant_histr[31][0], '32' : quant_histr[32][0], '33' : quant_histr[33][0], '34' : quant_histr[34][0], 
                '35' : quant_histr[35][0], '36' : quant_histr[36][0], '37' : quant_histr[37][0], '38' : quant_histr[38][0], '39' : quant_histr[39][0],
                '40' : quant_histr[40][0], '41' : quant_histr[41][0], '42' : quant_histr[42][0], '43' : quant_histr[43][0], '44' : quant_histr[44][0], 
                '45' : quant_histr[45][0], '46' : quant_histr[46][0], '47' : quant_histr[47][0], '48' : quant_histr[48][0], '49' : quant_histr[49][0], 
                '50' : quant_histr[50][0], '51' : quant_histr[51][0], '52' : quant_histr[52][0], '53' : quant_histr[53][0], '54' : quant_histr[54][0], 
                '55' : quant_histr[55][0], '56' : quant_histr[56][0], '57' : quant_histr[57][0], '58' : quant_histr[58][0], '59' : quant_histr[59][0], 
                '60' : quant_histr[60][0], '61' : quant_histr[61][0], '62' : quant_histr[62][0], '63' : quant_histr[63][0], '64' : quant_histr[64][0], 
                '65' : quant_histr[65][0], '66' : quant_histr[66][0], '67' : quant_histr[67][0], '68' : quant_histr[68][0], '69' : quant_histr[69][0], 
                '70' : quant_histr[70][0], '71' : quant_histr[71][0], '72' : quant_histr[72][0], '73' : quant_histr[73][0], '74' : quant_histr[74][0], 
                '75' : quant_histr[75][0], '76' : quant_histr[76][0], '77' : quant_histr[77][0], '78' : quant_histr[78][0], '79' : quant_histr[79][0], 
                '80' : quant_histr[80][0], '81' : quant_histr[81][0], '82' : quant_histr[82][0], '83' : quant_histr[83][0], '84' : quant_histr[84][0], 
                '85' : quant_histr[85][0], '86' : quant_histr[86][0], '87' : quant_histr[87][0], '88' : quant_histr[88][0], '89' : quant_histr[89][0], 
                '90' : quant_histr[90][0], '91' : quant_histr[91][0], '92' : quant_histr[92][0], '93' : quant_histr[93][0], '94' : quant_histr[94][0], 
                '95' : quant_histr[95][0], '96' : quant_histr[96][0], '97' : quant_histr[97][0], '98' : quant_histr[98][0], '99' : quant_histr[99][0], 
                '100' : quant_histr[100][0], '101' : quant_histr[101][0], '102' : quant_histr[102][0], '103' : quant_histr[103][0], '104' : quant_histr[104][0], 
                '105' : quant_histr[105][0], '106' : quant_histr[106][0], '107' : quant_histr[107][0], '108' : quant_histr[108][0], '109' : quant_histr[109][0], 
                '110' : quant_histr[110][0], '111' : quant_histr[111][0], '112' : quant_histr[112][0], '113' : quant_histr[113][0], '114' : quant_histr[114][0], 
                '115' : quant_histr[115][0], '116' : quant_histr[116][0], '117' : quant_histr[117][0], '118' : quant_histr[118][0], '119' : quant_histr[120][0],
                '120' : quant_histr[120][0], '121' : quant_histr[121][0], '122' : quant_histr[122][0], '123' : quant_histr[123][0], '124' : quant_histr[124][0], 
                '125' : quant_histr[125][0], '126' : quant_histr[126][0], '127' : quant_histr[127][0], '128' : quant_histr[128][0], '129' : quant_histr[129][0],
                '130' : quant_histr[130][0], '131' : quant_histr[131][0], '132' : quant_histr[132][0], '133' : quant_histr[133][0], '134' : quant_histr[134][0], 
                '135' : quant_histr[135][0], '136' : quant_histr[136][0], '137' : quant_histr[137][0], '138' : quant_histr[138][0], '139' : quant_histr[139][0],
                '140' : quant_histr[140][0], '141' : quant_histr[141][0], '142' : quant_histr[142][0], '143' : quant_histr[143][0], '144' : quant_histr[144][0], 
                '145' : quant_histr[145][0], '146' : quant_histr[146][0], '147' : quant_histr[147][0], '148' : quant_histr[148][0], '149' : quant_histr[149][0], 
                '150' : quant_histr[150][0], '151' : quant_histr[151][0], '152' : quant_histr[152][0], '153' : quant_histr[153][0], '154' : quant_histr[154][0], 
                '155' : quant_histr[155][0], '156' : quant_histr[156][0], '157' : quant_histr[157][0], '158' : quant_histr[158][0], '159' : quant_histr[159][0], 
                '160' : quant_histr[160][0], '161' : quant_histr[161][0], '162' : quant_histr[162][0], '163' : quant_histr[163][0], '164' : quant_histr[164][0], 
                '165' : quant_histr[165][0], '166' : quant_histr[166][0], '167' : quant_histr[167][0], '168' : quant_histr[168][0], '169' : quant_histr[169][0], 
                '170' : quant_histr[170][0], '171' : quant_histr[171][0], '172' : quant_histr[172][0], '173' : quant_histr[173][0], '174' : quant_histr[174][0], 
                '175' : quant_histr[175][0], '176' : quant_histr[176][0], '177' : quant_histr[177][0], '178' : quant_histr[178][0], '179' : quant_histr[179][0], 
                '180' : quant_histr[180][0], '181' : quant_histr[181][0], '182' : quant_histr[182][0], '183' : quant_histr[183][0], '184' : quant_histr[184][0], 
                '185' : quant_histr[185][0], '186' : quant_histr[186][0], '187' : quant_histr[187][0], '188' : quant_histr[188][0], '189' : quant_histr[189][0], 
                '190' : quant_histr[190][0], '191' : quant_histr[191][0], '192' : quant_histr[192][0], '193' : quant_histr[193][0], '194' : quant_histr[194][0], 
                '195' : quant_histr[195][0], '196' : quant_histr[196][0], '197' : quant_histr[197][0], '198' : quant_histr[198][0], '199' : quant_histr[199][0],
                '200' : quant_histr[200][0], '201' : quant_histr[201][0], '202' : quant_histr[202][0], '203' : quant_histr[203][0], '204' : quant_histr[204][0], 
                '205' : quant_histr[205][0], '206' : quant_histr[206][0], '207' : quant_histr[207][0], '208' : quant_histr[208][0], '209' : quant_histr[209][0], 
                '210' : quant_histr[210][0], '211' : quant_histr[211][0], '212' : quant_histr[212][0], '213' : quant_histr[213][0], '214' : quant_histr[214][0], 
                '215' : quant_histr[215][0], '216' : quant_histr[216][0], '217' : quant_histr[217][0], '218' : quant_histr[218][0], '219' : quant_histr[220][0],
                '220' : quant_histr[220][0], '221' : quant_histr[221][0], '222' : quant_histr[222][0], '223' : quant_histr[223][0], '224' : quant_histr[224][0], 
                '225' : quant_histr[225][0], '226' : quant_histr[226][0], '227' : quant_histr[227][0], '228' : quant_histr[228][0], '229' : quant_histr[229][0],
                '230' : quant_histr[230][0], '231' : quant_histr[231][0], '232' : quant_histr[232][0], '233' : quant_histr[233][0], '234' : quant_histr[234][0], 
                '235' : quant_histr[235][0], '236' : quant_histr[236][0], '237' : quant_histr[237][0], '238' : quant_histr[238][0], '239' : quant_histr[239][0],
                '240' : quant_histr[240][0], '241' : quant_histr[241][0], '242' : quant_histr[242][0], '243' : quant_histr[243][0], '244' : quant_histr[244][0], 
                '245' : quant_histr[245][0], '246' : quant_histr[246][0], '247' : quant_histr[247][0], '248' : quant_histr[248][0], '249' : quant_histr[249][0], 
                '250' : quant_histr[250][0], '251' : quant_histr[251][0], '252' : quant_histr[252][0], '253' : quant_histr[253][0], '254' : quant_histr[254][0], 
                '255' : quant_histr[255][0],
                'label': label_dir}
                
                new_g_row = {'filename': imagename,
                '0' : gray_histr[0][0], '1' : gray_histr[1][0], '2' : gray_histr[2][0], '3' : gray_histr[3][0], '4' : gray_histr[4][0], 
                '5' : gray_histr[5][0], '6' : gray_histr[6][0], '7' : gray_histr[7][0], '8' : gray_histr[8][0], '9' : gray_histr[9][0], 
                '10' : gray_histr[10][0], '11' : gray_histr[11][0], '12' : gray_histr[12][0], '13' : gray_histr[13][0], '14' : gray_histr[14][0], 
                '15' : gray_histr[15][0], '16' : gray_histr[16][0], '17' : gray_histr[17][0], '18' : gray_histr[18][0], '19' : gray_histr[20][0],
                '20' : gray_histr[20][0], '21' : gray_histr[21][0], '22' : gray_histr[22][0], '23' : gray_histr[23][0], '24' : gray_histr[24][0], 
                '25' : gray_histr[25][0], '26' : gray_histr[26][0], '27' : gray_histr[27][0], '28' : gray_histr[28][0], '29' : gray_histr[29][0],
                '30' : gray_histr[30][0], '31' : gray_histr[31][0], '32' : gray_histr[32][0], '33' : gray_histr[33][0], '34' : gray_histr[34][0], 
                '35' : gray_histr[35][0], '36' : gray_histr[36][0], '37' : gray_histr[37][0], '38' : gray_histr[38][0], '39' : gray_histr[39][0],
                '40' : gray_histr[40][0], '41' : gray_histr[41][0], '42' : gray_histr[42][0], '43' : gray_histr[43][0], '44' : gray_histr[44][0], 
                '45' : gray_histr[45][0], '46' : gray_histr[46][0], '47' : gray_histr[47][0], '48' : gray_histr[48][0], '49' : gray_histr[49][0], 
                '50' : gray_histr[50][0], '51' : gray_histr[51][0], '52' : gray_histr[52][0], '53' : gray_histr[53][0], '54' : gray_histr[54][0], 
                '55' : gray_histr[55][0], '56' : gray_histr[56][0], '57' : gray_histr[57][0], '58' : gray_histr[58][0], '59' : gray_histr[59][0], 
                '60' : gray_histr[60][0], '61' : gray_histr[61][0], '62' : gray_histr[62][0], '63' : gray_histr[63][0], '64' : gray_histr[64][0], 
                '65' : gray_histr[65][0], '66' : gray_histr[66][0], '67' : gray_histr[67][0], '68' : gray_histr[68][0], '69' : gray_histr[69][0], 
                '70' : gray_histr[70][0], '71' : gray_histr[71][0], '72' : gray_histr[72][0], '73' : gray_histr[73][0], '74' : gray_histr[74][0], 
                '75' : gray_histr[75][0], '76' : gray_histr[76][0], '77' : gray_histr[77][0], '78' : gray_histr[78][0], '79' : gray_histr[79][0], 
                '80' : gray_histr[80][0], '81' : gray_histr[81][0], '82' : gray_histr[82][0], '83' : gray_histr[83][0], '84' : gray_histr[84][0], 
                '85' : gray_histr[85][0], '86' : gray_histr[86][0], '87' : gray_histr[87][0], '88' : gray_histr[88][0], '89' : gray_histr[89][0], 
                '90' : gray_histr[90][0], '91' : gray_histr[91][0], '92' : gray_histr[92][0], '93' : gray_histr[93][0], '94' : gray_histr[94][0], 
                '95' : gray_histr[95][0], '96' : gray_histr[96][0], '97' : gray_histr[97][0], '98' : gray_histr[98][0], '99' : gray_histr[99][0], 
                '100' : gray_histr[100][0], '101' : gray_histr[101][0], '102' : gray_histr[102][0], '103' : gray_histr[103][0], '104' : gray_histr[104][0], 
                '105' : gray_histr[105][0], '106' : gray_histr[106][0], '107' : gray_histr[107][0], '108' : gray_histr[108][0], '109' : gray_histr[109][0], 
                '110' : gray_histr[110][0], '111' : gray_histr[111][0], '112' : gray_histr[112][0], '113' : gray_histr[113][0], '114' : gray_histr[114][0], 
                '115' : gray_histr[115][0], '116' : gray_histr[116][0], '117' : gray_histr[117][0], '118' : gray_histr[118][0], '119' : gray_histr[120][0],
                '120' : gray_histr[120][0], '121' : gray_histr[121][0], '122' : gray_histr[122][0], '123' : gray_histr[123][0], '124' : gray_histr[124][0], 
                '125' : gray_histr[125][0], '126' : gray_histr[126][0], '127' : gray_histr[127][0], '128' : gray_histr[128][0], '129' : gray_histr[129][0],
                '130' : gray_histr[130][0], '131' : gray_histr[131][0], '132' : gray_histr[132][0], '133' : gray_histr[133][0], '134' : gray_histr[134][0], 
                '135' : gray_histr[135][0], '136' : gray_histr[136][0], '137' : gray_histr[137][0], '138' : gray_histr[138][0], '139' : gray_histr[139][0],
                '140' : gray_histr[140][0], '141' : gray_histr[141][0], '142' : gray_histr[142][0], '143' : gray_histr[143][0], '144' : gray_histr[144][0], 
                '145' : gray_histr[145][0], '146' : gray_histr[146][0], '147' : gray_histr[147][0], '148' : gray_histr[148][0], '149' : gray_histr[149][0], 
                '150' : gray_histr[150][0], '151' : gray_histr[151][0], '152' : gray_histr[152][0], '153' : gray_histr[153][0], '154' : gray_histr[154][0], 
                '155' : gray_histr[155][0], '156' : gray_histr[156][0], '157' : gray_histr[157][0], '158' : gray_histr[158][0], '159' : gray_histr[159][0], 
                '160' : gray_histr[160][0], '161' : gray_histr[161][0], '162' : gray_histr[162][0], '163' : gray_histr[163][0], '164' : gray_histr[164][0], 
                '165' : gray_histr[165][0], '166' : gray_histr[166][0], '167' : gray_histr[167][0], '168' : gray_histr[168][0], '169' : gray_histr[169][0], 
                '170' : gray_histr[170][0], '171' : gray_histr[171][0], '172' : gray_histr[172][0], '173' : gray_histr[173][0], '174' : gray_histr[174][0], 
                '175' : gray_histr[175][0], '176' : gray_histr[176][0], '177' : gray_histr[177][0], '178' : gray_histr[178][0], '179' : gray_histr[179][0], 
                '180' : gray_histr[180][0], '181' : gray_histr[181][0], '182' : gray_histr[182][0], '183' : gray_histr[183][0], '184' : gray_histr[184][0], 
                '185' : gray_histr[185][0], '186' : gray_histr[186][0], '187' : gray_histr[187][0], '188' : gray_histr[188][0], '189' : gray_histr[189][0], 
                '190' : gray_histr[190][0], '191' : gray_histr[191][0], '192' : gray_histr[192][0], '193' : gray_histr[193][0], '194' : gray_histr[194][0], 
                '195' : gray_histr[195][0], '196' : gray_histr[196][0], '197' : gray_histr[197][0], '198' : gray_histr[198][0], '199' : gray_histr[199][0],
                '200' : gray_histr[200][0], '201' : gray_histr[201][0], '202' : gray_histr[202][0], '203' : gray_histr[203][0], '204' : gray_histr[204][0], 
                '205' : gray_histr[205][0], '206' : gray_histr[206][0], '207' : gray_histr[207][0], '208' : gray_histr[208][0], '209' : gray_histr[209][0], 
                '210' : gray_histr[210][0], '211' : gray_histr[211][0], '212' : gray_histr[212][0], '213' : gray_histr[213][0], '214' : gray_histr[214][0], 
                '215' : gray_histr[215][0], '216' : gray_histr[216][0], '217' : gray_histr[217][0], '218' : gray_histr[218][0], '219' : gray_histr[220][0],
                '220' : gray_histr[220][0], '221' : gray_histr[221][0], '222' : gray_histr[222][0], '223' : gray_histr[223][0], '224' : gray_histr[224][0], 
                '225' : gray_histr[225][0], '226' : gray_histr[226][0], '227' : gray_histr[227][0], '228' : gray_histr[228][0], '229' : gray_histr[229][0],
                '230' : gray_histr[230][0], '231' : gray_histr[231][0], '232' : gray_histr[232][0], '233' : gray_histr[233][0], '234' : gray_histr[234][0], 
                '235' : gray_histr[235][0], '236' : gray_histr[236][0], '237' : gray_histr[237][0], '238' : gray_histr[238][0], '239' : gray_histr[239][0],
                '240' : gray_histr[240][0], '241' : gray_histr[241][0], '242' : gray_histr[242][0], '243' : gray_histr[243][0], '244' : gray_histr[244][0], 
                '245' : gray_histr[245][0], '246' : gray_histr[246][0], '247' : gray_histr[247][0], '248' : gray_histr[248][0], '249' : gray_histr[249][0], 
                '250' : gray_histr[250][0], '251' : gray_histr[251][0], '252' : gray_histr[252][0], '253' : gray_histr[253][0], '254' : gray_histr[254][0], 
                '255' : gray_histr[255][0],
                'label': label_dir}
                my_e_df = my_e_df.append(new_e_row, ignore_index=True)
                my_q_df = my_q_df.append(new_q_row, ignore_index=True)
                my_g_df = my_g_df.append(new_g_row, ignore_index=True)
            except:
                new_row = {'filename': imagename,
                '0' : 0.0, '1' : 0.0, '2' : 0.0, '3' : 0.0, '4' : 0.0, '5' : 0.0, '6' : 0.0, '7' : 0.0, '8' : 0.0, '9' : 0.0, 
                '10' : 0.0, '11' : 0.0, '12' : 0.0, '13' : 0.0, '14' : 0.0, '15' : 0.0, '16' : 0.0, '17' : 0.0, '18' : 0.0, '19' : 0.0,
                '20' : 0.0, '21' : 0.0, '22' : 0.0, '23' : 0.0, '24' : 0.0, '25' : 0.0, '26' : 0.0, '27' : 0.0, '28' : 0.0, '29' : 0.0,
                '30' : 0.0, '31' : 0.0, '32' : 0.0, '33' : 0.0, '34' : 0.0, '35' : 0.0, '36' : 0.0, '37' : 0.0, '38' : 0.0, '39' : 0.0,
                '40' : 0.0, '41' : 0.0, '42' : 0.0, '43' : 0.0, '44' : 0.0, '45' : 0.0, '46' : 0.0, '47' : 0.0, '48' : 0.0, '49' : 0.0,
                '50' : 0.0, '51' : 0.0, '52' : 0.0, '53' : 0.0, '54' : 0.0, '55' : 0.0, '56' : 0.0, '57' : 0.0, '58' : 0.0, '59' : 0.0,
                '60' : 0.0, '61' : 0.0, '62' : 0.0, '63' : 0.0, '64' : 0.0, '65' : 0.0, '66' : 0.0, '67' : 0.0, '68' : 0.0, '69' : 0.0, 
                '70' : 0.0, '71' : 0.0, '72' : 0.0, '73' : 0.0, '74' : 0.0, '75' : 0.0, '76' : 0.0, '77' : 0.0, '78' : 0.0, '79' : 0.0, 
                '80' : 0.0, '81' : 0.0, '82' : 0.0, '83' : 0.0, '84' : 0.0, '85' : 0.0, '86' : 0.0, '87' : 0.0, '88' : 0.0, '89' : 0.0,
                '90' : 0.0, '91' : 0.0, '92' : 0.0, '93' : 0.0, '94' : 0.0, '95' : 0.0, '96' : 0.0, '97' : 0.0, '98' : 0.0, '99' : 0.0,
                '100' : 0.0, '101' : 0.0, '102' : 0.0, '103' : 0.0, '104' : 0.0, '105' : 0.0, '106' : 0.0, '107' : 0.0, '108' : 0.0, '109' : 0.0, 
                '110' : 0.0, '111' : 0.0, '112' : 0.0, '113' : 0.0, '114' : 0.0, '115' : 0.0, '116' : 0.0, '117' : 0.0, '118' : 0.0, '119' : 0.0,
                '120' : 0.0, '121' : 0.0, '122' : 0.0, '123' : 0.0, '124' : 0.0, '125' : 0.0, '126' : 0.0, '127' : 0.0, '128' : 0.0, '129' : 0.0,
                '130' : 0.0, '131' : 0.0, '132' : 0.0, '133' : 0.0, '134' : 0.0, '135' : 0.0, '136' : 0.0, '137' : 0.0, '138' : 0.0, '139' : 0.0,
                '140' : 0.0, '141' : 0.0, '142' : 0.0, '143' : 0.0, '144' : 0.0, '145' : 0.0, '146' : 0.0, '147' : 0.0, '148' : 0.0, '149' : 0.0,
                '150' : 0.0, '151' : 0.0, '152' : 0.0, '153' : 0.0, '154' : 0.0, '155' : 0.0, '156' : 0.0, '157' : 0.0, '158' : 0.0, '159' : 0.0,
                '160' : 0.0, '161' : 0.0, '162' : 0.0, '163' : 0.0, '164' : 0.0, '165' : 0.0, '166' : 0.0, '167' : 0.0, '168' : 0.0, '169' : 0.0, 
                '170' : 0.0, '171' : 0.0, '172' : 0.0, '173' : 0.0, '174' : 0.0, '175' : 0.0, '176' : 0.0, '177' : 0.0, '178' : 0.0, '179' : 0.0, 
                '180' : 0.0, '181' : 0.0, '182' : 0.0, '183' : 0.0, '184' : 0.0, '185' : 0.0, '186' : 0.0, '187' : 0.0, '188' : 0.0, '189' : 0.0,
                '190' : 0.0, '191' : 0.0, '192' : 0.0, '193' : 0.0, '194' : 0.0, '195' : 0.0, '196' : 0.0, '197' : 0.0, '198' : 0.0, '199' : 0.0,
                '200' : 0.0, '201' : 0.0, '202' : 0.0, '203' : 0.0, '204' : 0.0, '205' : 0.0, '206' : 0.0, '207' : 0.0, '208' : 0.0, '209' : 0.0, 
                '210' : 0.0, '211' : 0.0, '212' : 0.0, '213' : 0.0, '214' : 0.0, '215' : 0.0, '216' : 0.0, '217' : 0.0, '218' : 0.0, '219' : 0.0,
                '220' : 0.0, '221' : 0.0, '222' : 0.0, '223' : 0.0, '224' : 0.0, '225' : 0.0, '226' : 0.0, '227' : 0.0, '228' : 0.0, '229' : 0.0,
                '230' : 0.0, '231' : 0.0, '232' : 0.0, '233' : 0.0, '234' : 0.0, '235' : 0.0, '236' : 0.0, '237' : 0.0, '238' : 0.0, '239' : 0.0,
                '240' : 0.0, '241' : 0.0, '242' : 0.0, '243' : 0.0, '244' : 0.0, '245' : 0.0, '246' : 0.0, '247' : 0.0, '248' : 0.0, '249' : 0.0,
                '250' : 0.0, '251' : 0.0, '252' : 0.0, '253' : 0.0, '254' : 0.0, '255' : 0.0,
                'label': label_dir}
                my_e_df = my_e_df.append(new_row, ignore_index=True)
                my_q_df = my_q_df.append(new_row, ignore_index=True)
                my_g_df = my_g_df.append(new_row, ignore_index=True)

my_e_df.to_csv('histograms_e.csv', index=False)
my_q_df.to_csv('histograms_q.csv', index=False)
my_g_df.to_csv('histograms_g.csv', index=False)
