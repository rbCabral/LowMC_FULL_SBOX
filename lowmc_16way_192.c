/*! @file lowmc_16way_192.c
 *  @brief 16-way optimized implementation of the LowMC-192-192-4   
 *  The code is provided under the GPL license, see LICENSE for
 *  more details.
  */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "immintrin.h"
#include "emmintrin.h"
#include "constants.h"


#include "bench.h"


#define BENCH 1000


#define PARALLEL    8
#define ROUNDS       4

#define loadLinha(i)            resp =  _mm256_setzero_si256();\
                                for(j=0;j<12;j++){\
                                    temp = _mm256_loadu_si256((__m256i*)matrix+(i*12+j));\
                                    temp = _mm256_and_si256(temp,r[j]);\
                                    resp = _mm256_xor_si256(resp,temp);\
                                }


uint64_t transpose8(uint64_t x) {

    x = (x & 0xAA55AA55AA55AA55LL) | ((x & 0x00AA00AA00AA00AALL) << 7)  | ((x >> 7) & 0x00AA00AA00AA00AALL);
    x = (x & 0xCCCC3333CCCC3333LL) | ((x & 0x0000CCCC0000CCCCLL) << 14) | ((x >> 14) & 0x0000CCCC0000CCCCLL);
    x = (x & 0xF0F0F0F00F0F0F0FLL) | ((x & 0x00000000F0F0F0F0LL) << 28) | ((x >> 28) & 0x00000000F0F0F0F0LL);
    
    return x;

} 


void stateTransBack(uint8_t in[][24], uint8_t out[][24]){
    int i,j,k;
    uint64_t x;
    int t=0;
    for(i=0;i<24;i++){
        x = 0;
        for(j=0;j<8;j++){
            x = x << 8 | in[j][i];  
        }
        x = transpose8(x);
        for (k = t+7; k >= t; k--) { 
            out[k][i] = x; x = x >> 8;
        }
    }
}


void stateTrans(uint8_t in[][24], uint8_t out[][24]){
    int j,k;
    
    uint64_t x = 0;
    for(j=0;j<24;j++){
        for(k=0;k<8;k++){
            x = x << 8 | in[k][j];  
        }
        x = transpose8(x);
        for (k = 7; k >= 0; k--) { 
            out[k][j] = x; x = x >> 8;
        }
    } 
}

void stateTrans2(uint8_t in[][24], uint8_t out[][16]){
out[0][0] = in[0][0];
out[16][0] = in[0][1];
out[8][1] = in[0][2];
out[0][2] = in[0][3];
out[16][2] = in[0][4];
out[8][3] = in[0][5];
out[0][4] = in[0][6];
out[16][4] = in[0][7];
out[8][5] = in[0][8];
out[0][6] = in[0][9];
out[16][6] = in[0][10];
out[8][7] = in[0][11];
out[0][8] = in[0][12];
out[16][8] = in[0][13];
out[8][9] = in[0][14];
out[0][10] = in[0][15];
out[16][10] = in[0][16];
out[8][11] = in[0][17];
out[0][12] = in[0][18];
out[16][12] = in[0][19];
out[8][13] = in[0][20];
out[0][14] = in[0][21];
out[16][14] = in[0][22];
out[8][15] = in[0][23];
out[1][0] = in[8][0];
out[17][0] = in[8][1];
out[9][1] = in[8][2];
out[1][2] = in[8][3];
out[17][2] = in[8][4];
out[9][3] = in[8][5];
out[1][4] = in[8][6];
out[17][4] = in[8][7];
out[9][5] = in[8][8];
out[1][6] = in[8][9];
out[17][6] = in[8][10];
out[9][7] = in[8][11];
out[1][8] = in[8][12];
out[17][8] = in[8][13];
out[9][9] = in[8][14];
out[1][10] = in[8][15];
out[17][10] = in[8][16];
out[9][11] = in[8][17];
out[1][12] = in[8][18];
out[17][12] = in[8][19];
out[9][13] = in[8][20];
out[1][14] = in[8][21];
out[17][14] = in[8][22];
out[9][15] = in[8][23];
out[2][0] = in[1][0];
out[18][0] = in[1][1];
out[10][1] = in[1][2];
out[2][2] = in[1][3];
out[18][2] = in[1][4];
out[10][3] = in[1][5];
out[2][4] = in[1][6];
out[18][4] = in[1][7];
out[10][5] = in[1][8];
out[2][6] = in[1][9];
out[18][6] = in[1][10];
out[10][7] = in[1][11];
out[2][8] = in[1][12];
out[18][8] = in[1][13];
out[10][9] = in[1][14];
out[2][10] = in[1][15];
out[18][10] = in[1][16];
out[10][11] = in[1][17];
out[2][12] = in[1][18];
out[18][12] = in[1][19];
out[10][13] = in[1][20];
out[2][14] = in[1][21];
out[18][14] = in[1][22];
out[10][15] = in[1][23];
out[3][0] = in[9][0];
out[19][0] = in[9][1];
out[11][1] = in[9][2];
out[3][2] = in[9][3];
out[19][2] = in[9][4];
out[11][3] = in[9][5];
out[3][4] = in[9][6];
out[19][4] = in[9][7];
out[11][5] = in[9][8];
out[3][6] = in[9][9];
out[19][6] = in[9][10];
out[11][7] = in[9][11];
out[3][8] = in[9][12];
out[19][8] = in[9][13];
out[11][9] = in[9][14];
out[3][10] = in[9][15];
out[19][10] = in[9][16];
out[11][11] = in[9][17];
out[3][12] = in[9][18];
out[19][12] = in[9][19];
out[11][13] = in[9][20];
out[3][14] = in[9][21];
out[19][14] = in[9][22];
out[11][15] = in[9][23];
out[4][0] = in[2][0];
out[20][0] = in[2][1];
out[12][1] = in[2][2];
out[4][2] = in[2][3];
out[20][2] = in[2][4];
out[12][3] = in[2][5];
out[4][4] = in[2][6];
out[20][4] = in[2][7];
out[12][5] = in[2][8];
out[4][6] = in[2][9];
out[20][6] = in[2][10];
out[12][7] = in[2][11];
out[4][8] = in[2][12];
out[20][8] = in[2][13];
out[12][9] = in[2][14];
out[4][10] = in[2][15];
out[20][10] = in[2][16];
out[12][11] = in[2][17];
out[4][12] = in[2][18];
out[20][12] = in[2][19];
out[12][13] = in[2][20];
out[4][14] = in[2][21];
out[20][14] = in[2][22];
out[12][15] = in[2][23];
out[5][0] = in[10][0];
out[21][0] = in[10][1];
out[13][1] = in[10][2];
out[5][2] = in[10][3];
out[21][2] = in[10][4];
out[13][3] = in[10][5];
out[5][4] = in[10][6];
out[21][4] = in[10][7];
out[13][5] = in[10][8];
out[5][6] = in[10][9];
out[21][6] = in[10][10];
out[13][7] = in[10][11];
out[5][8] = in[10][12];
out[21][8] = in[10][13];
out[13][9] = in[10][14];
out[5][10] = in[10][15];
out[21][10] = in[10][16];
out[13][11] = in[10][17];
out[5][12] = in[10][18];
out[21][12] = in[10][19];
out[13][13] = in[10][20];
out[5][14] = in[10][21];
out[21][14] = in[10][22];
out[13][15] = in[10][23];
out[6][0] = in[3][0];
out[22][0] = in[3][1];
out[14][1] = in[3][2];
out[6][2] = in[3][3];
out[22][2] = in[3][4];
out[14][3] = in[3][5];
out[6][4] = in[3][6];
out[22][4] = in[3][7];
out[14][5] = in[3][8];
out[6][6] = in[3][9];
out[22][6] = in[3][10];
out[14][7] = in[3][11];
out[6][8] = in[3][12];
out[22][8] = in[3][13];
out[14][9] = in[3][14];
out[6][10] = in[3][15];
out[22][10] = in[3][16];
out[14][11] = in[3][17];
out[6][12] = in[3][18];
out[22][12] = in[3][19];
out[14][13] = in[3][20];
out[6][14] = in[3][21];
out[22][14] = in[3][22];
out[14][15] = in[3][23];
out[7][0] = in[11][0];
out[23][0] = in[11][1];
out[15][1] = in[11][2];
out[7][2] = in[11][3];
out[23][2] = in[11][4];
out[15][3] = in[11][5];
out[7][4] = in[11][6];
out[23][4] = in[11][7];
out[15][5] = in[11][8];
out[7][6] = in[11][9];
out[23][6] = in[11][10];
out[15][7] = in[11][11];
out[7][8] = in[11][12];
out[23][8] = in[11][13];
out[15][9] = in[11][14];
out[7][10] = in[11][15];
out[23][10] = in[11][16];
out[15][11] = in[11][17];
out[7][12] = in[11][18];
out[23][12] = in[11][19];
out[15][13] = in[11][20];
out[7][14] = in[11][21];
out[23][14] = in[11][22];
out[15][15] = in[11][23];
out[8][0] = in[4][0];
out[0][1] = in[4][1];
out[16][1] = in[4][2];
out[8][2] = in[4][3];
out[0][3] = in[4][4];
out[16][3] = in[4][5];
out[8][4] = in[4][6];
out[0][5] = in[4][7];
out[16][5] = in[4][8];
out[8][6] = in[4][9];
out[0][7] = in[4][10];
out[16][7] = in[4][11];
out[8][8] = in[4][12];
out[0][9] = in[4][13];
out[16][9] = in[4][14];
out[8][10] = in[4][15];
out[0][11] = in[4][16];
out[16][11] = in[4][17];
out[8][12] = in[4][18];
out[0][13] = in[4][19];
out[16][13] = in[4][20];
out[8][14] = in[4][21];
out[0][15] = in[4][22];
out[16][15] = in[4][23];
out[9][0] = in[12][0];
out[1][1] = in[12][1];
out[17][1] = in[12][2];
out[9][2] = in[12][3];
out[1][3] = in[12][4];
out[17][3] = in[12][5];
out[9][4] = in[12][6];
out[1][5] = in[12][7];
out[17][5] = in[12][8];
out[9][6] = in[12][9];
out[1][7] = in[12][10];
out[17][7] = in[12][11];
out[9][8] = in[12][12];
out[1][9] = in[12][13];
out[17][9] = in[12][14];
out[9][10] = in[12][15];
out[1][11] = in[12][16];
out[17][11] = in[12][17];
out[9][12] = in[12][18];
out[1][13] = in[12][19];
out[17][13] = in[12][20];
out[9][14] = in[12][21];
out[1][15] = in[12][22];
out[17][15] = in[12][23];
out[10][0] = in[5][0];
out[2][1] = in[5][1];
out[18][1] = in[5][2];
out[10][2] = in[5][3];
out[2][3] = in[5][4];
out[18][3] = in[5][5];
out[10][4] = in[5][6];
out[2][5] = in[5][7];
out[18][5] = in[5][8];
out[10][6] = in[5][9];
out[2][7] = in[5][10];
out[18][7] = in[5][11];
out[10][8] = in[5][12];
out[2][9] = in[5][13];
out[18][9] = in[5][14];
out[10][10] = in[5][15];
out[2][11] = in[5][16];
out[18][11] = in[5][17];
out[10][12] = in[5][18];
out[2][13] = in[5][19];
out[18][13] = in[5][20];
out[10][14] = in[5][21];
out[2][15] = in[5][22];
out[18][15] = in[5][23];
out[11][0] = in[13][0];
out[3][1] = in[13][1];
out[19][1] = in[13][2];
out[11][2] = in[13][3];
out[3][3] = in[13][4];
out[19][3] = in[13][5];
out[11][4] = in[13][6];
out[3][5] = in[13][7];
out[19][5] = in[13][8];
out[11][6] = in[13][9];
out[3][7] = in[13][10];
out[19][7] = in[13][11];
out[11][8] = in[13][12];
out[3][9] = in[13][13];
out[19][9] = in[13][14];
out[11][10] = in[13][15];
out[3][11] = in[13][16];
out[19][11] = in[13][17];
out[11][12] = in[13][18];
out[3][13] = in[13][19];
out[19][13] = in[13][20];
out[11][14] = in[13][21];
out[3][15] = in[13][22];
out[19][15] = in[13][23];
out[12][0] = in[6][0];
out[4][1] = in[6][1];
out[20][1] = in[6][2];
out[12][2] = in[6][3];
out[4][3] = in[6][4];
out[20][3] = in[6][5];
out[12][4] = in[6][6];
out[4][5] = in[6][7];
out[20][5] = in[6][8];
out[12][6] = in[6][9];
out[4][7] = in[6][10];
out[20][7] = in[6][11];
out[12][8] = in[6][12];
out[4][9] = in[6][13];
out[20][9] = in[6][14];
out[12][10] = in[6][15];
out[4][11] = in[6][16];
out[20][11] = in[6][17];
out[12][12] = in[6][18];
out[4][13] = in[6][19];
out[20][13] = in[6][20];
out[12][14] = in[6][21];
out[4][15] = in[6][22];
out[20][15] = in[6][23];
out[13][0] = in[14][0];
out[5][1] = in[14][1];
out[21][1] = in[14][2];
out[13][2] = in[14][3];
out[5][3] = in[14][4];
out[21][3] = in[14][5];
out[13][4] = in[14][6];
out[5][5] = in[14][7];
out[21][5] = in[14][8];
out[13][6] = in[14][9];
out[5][7] = in[14][10];
out[21][7] = in[14][11];
out[13][8] = in[14][12];
out[5][9] = in[14][13];
out[21][9] = in[14][14];
out[13][10] = in[14][15];
out[5][11] = in[14][16];
out[21][11] = in[14][17];
out[13][12] = in[14][18];
out[5][13] = in[14][19];
out[21][13] = in[14][20];
out[13][14] = in[14][21];
out[5][15] = in[14][22];
out[21][15] = in[14][23];
out[14][0] = in[7][0];
out[6][1] = in[7][1];
out[22][1] = in[7][2];
out[14][2] = in[7][3];
out[6][3] = in[7][4];
out[22][3] = in[7][5];
out[14][4] = in[7][6];
out[6][5] = in[7][7];
out[22][5] = in[7][8];
out[14][6] = in[7][9];
out[6][7] = in[7][10];
out[22][7] = in[7][11];
out[14][8] = in[7][12];
out[6][9] = in[7][13];
out[22][9] = in[7][14];
out[14][10] = in[7][15];
out[6][11] = in[7][16];
out[22][11] = in[7][17];
out[14][12] = in[7][18];
out[6][13] = in[7][19];
out[22][13] = in[7][20];
out[14][14] = in[7][21];
out[6][15] = in[7][22];
out[22][15] = in[7][23];
out[15][0] = in[15][0];
out[7][1] = in[15][1];
out[23][1] = in[15][2];
out[15][2] = in[15][3];
out[7][3] = in[15][4];
out[23][3] = in[15][5];
out[15][4] = in[15][6];
out[7][5] = in[15][7];
out[23][5] = in[15][8];
out[15][6] = in[15][9];
out[7][7] = in[15][10];
out[23][7] = in[15][11];
out[15][8] = in[15][12];
out[7][9] = in[15][13];
out[23][9] = in[15][14];
out[15][10] = in[15][15];
out[7][11] = in[15][16];
out[23][11] = in[15][17];
out[15][12] = in[15][18];
out[7][13] = in[15][19];
out[23][13] = in[15][20];
out[15][14] = in[15][21];
out[7][15] = in[15][22];
out[23][15] = in[15][23];
}

void stateTrans2Back(uint8_t in[][16], uint8_t out[][24]){
out[0][0] = in[0][0];
out[4][1] = in[0][1];
out[0][3] = in[0][2];
out[4][4] = in[0][3];
out[0][6] = in[0][4];
out[4][7] = in[0][5];
out[0][9] = in[0][6];
out[4][10] = in[0][7];
out[0][12] = in[0][8];
out[4][13] = in[0][9];
out[0][15] = in[0][10];
out[4][16] = in[0][11];
out[0][18] = in[0][12];
out[4][19] = in[0][13];
out[0][21] = in[0][14];
out[4][22] = in[0][15];
out[8][0] = in[1][0];
out[12][1] = in[1][1];
out[8][3] = in[1][2];
out[12][4] = in[1][3];
out[8][6] = in[1][4];
out[12][7] = in[1][5];
out[8][9] = in[1][6];
out[12][10] = in[1][7];
out[8][12] = in[1][8];
out[12][13] = in[1][9];
out[8][15] = in[1][10];
out[12][16] = in[1][11];
out[8][18] = in[1][12];
out[12][19] = in[1][13];
out[8][21] = in[1][14];
out[12][22] = in[1][15];
out[1][0] = in[2][0];
out[5][1] = in[2][1];
out[1][3] = in[2][2];
out[5][4] = in[2][3];
out[1][6] = in[2][4];
out[5][7] = in[2][5];
out[1][9] = in[2][6];
out[5][10] = in[2][7];
out[1][12] = in[2][8];
out[5][13] = in[2][9];
out[1][15] = in[2][10];
out[5][16] = in[2][11];
out[1][18] = in[2][12];
out[5][19] = in[2][13];
out[1][21] = in[2][14];
out[5][22] = in[2][15];
out[9][0] = in[3][0];
out[13][1] = in[3][1];
out[9][3] = in[3][2];
out[13][4] = in[3][3];
out[9][6] = in[3][4];
out[13][7] = in[3][5];
out[9][9] = in[3][6];
out[13][10] = in[3][7];
out[9][12] = in[3][8];
out[13][13] = in[3][9];
out[9][15] = in[3][10];
out[13][16] = in[3][11];
out[9][18] = in[3][12];
out[13][19] = in[3][13];
out[9][21] = in[3][14];
out[13][22] = in[3][15];
out[2][0] = in[4][0];
out[6][1] = in[4][1];
out[2][3] = in[4][2];
out[6][4] = in[4][3];
out[2][6] = in[4][4];
out[6][7] = in[4][5];
out[2][9] = in[4][6];
out[6][10] = in[4][7];
out[2][12] = in[4][8];
out[6][13] = in[4][9];
out[2][15] = in[4][10];
out[6][16] = in[4][11];
out[2][18] = in[4][12];
out[6][19] = in[4][13];
out[2][21] = in[4][14];
out[6][22] = in[4][15];
out[10][0] = in[5][0];
out[14][1] = in[5][1];
out[10][3] = in[5][2];
out[14][4] = in[5][3];
out[10][6] = in[5][4];
out[14][7] = in[5][5];
out[10][9] = in[5][6];
out[14][10] = in[5][7];
out[10][12] = in[5][8];
out[14][13] = in[5][9];
out[10][15] = in[5][10];
out[14][16] = in[5][11];
out[10][18] = in[5][12];
out[14][19] = in[5][13];
out[10][21] = in[5][14];
out[14][22] = in[5][15];
out[3][0] = in[6][0];
out[7][1] = in[6][1];
out[3][3] = in[6][2];
out[7][4] = in[6][3];
out[3][6] = in[6][4];
out[7][7] = in[6][5];
out[3][9] = in[6][6];
out[7][10] = in[6][7];
out[3][12] = in[6][8];
out[7][13] = in[6][9];
out[3][15] = in[6][10];
out[7][16] = in[6][11];
out[3][18] = in[6][12];
out[7][19] = in[6][13];
out[3][21] = in[6][14];
out[7][22] = in[6][15];
out[11][0] = in[7][0];
out[15][1] = in[7][1];
out[11][3] = in[7][2];
out[15][4] = in[7][3];
out[11][6] = in[7][4];
out[15][7] = in[7][5];
out[11][9] = in[7][6];
out[15][10] = in[7][7];
out[11][12] = in[7][8];
out[15][13] = in[7][9];
out[11][15] = in[7][10];
out[15][16] = in[7][11];
out[11][18] = in[7][12];
out[15][19] = in[7][13];
out[11][21] = in[7][14];
out[15][22] = in[7][15];
out[4][0] = in[8][0];
out[0][2] = in[8][1];
out[4][3] = in[8][2];
out[0][5] = in[8][3];
out[4][6] = in[8][4];
out[0][8] = in[8][5];
out[4][9] = in[8][6];
out[0][11] = in[8][7];
out[4][12] = in[8][8];
out[0][14] = in[8][9];
out[4][15] = in[8][10];
out[0][17] = in[8][11];
out[4][18] = in[8][12];
out[0][20] = in[8][13];
out[4][21] = in[8][14];
out[0][23] = in[8][15];
out[12][0] = in[9][0];
out[8][2] = in[9][1];
out[12][3] = in[9][2];
out[8][5] = in[9][3];
out[12][6] = in[9][4];
out[8][8] = in[9][5];
out[12][9] = in[9][6];
out[8][11] = in[9][7];
out[12][12] = in[9][8];
out[8][14] = in[9][9];
out[12][15] = in[9][10];
out[8][17] = in[9][11];
out[12][18] = in[9][12];
out[8][20] = in[9][13];
out[12][21] = in[9][14];
out[8][23] = in[9][15];
out[5][0] = in[10][0];
out[1][2] = in[10][1];
out[5][3] = in[10][2];
out[1][5] = in[10][3];
out[5][6] = in[10][4];
out[1][8] = in[10][5];
out[5][9] = in[10][6];
out[1][11] = in[10][7];
out[5][12] = in[10][8];
out[1][14] = in[10][9];
out[5][15] = in[10][10];
out[1][17] = in[10][11];
out[5][18] = in[10][12];
out[1][20] = in[10][13];
out[5][21] = in[10][14];
out[1][23] = in[10][15];
out[13][0] = in[11][0];
out[9][2] = in[11][1];
out[13][3] = in[11][2];
out[9][5] = in[11][3];
out[13][6] = in[11][4];
out[9][8] = in[11][5];
out[13][9] = in[11][6];
out[9][11] = in[11][7];
out[13][12] = in[11][8];
out[9][14] = in[11][9];
out[13][15] = in[11][10];
out[9][17] = in[11][11];
out[13][18] = in[11][12];
out[9][20] = in[11][13];
out[13][21] = in[11][14];
out[9][23] = in[11][15];
out[6][0] = in[12][0];
out[2][2] = in[12][1];
out[6][3] = in[12][2];
out[2][5] = in[12][3];
out[6][6] = in[12][4];
out[2][8] = in[12][5];
out[6][9] = in[12][6];
out[2][11] = in[12][7];
out[6][12] = in[12][8];
out[2][14] = in[12][9];
out[6][15] = in[12][10];
out[2][17] = in[12][11];
out[6][18] = in[12][12];
out[2][20] = in[12][13];
out[6][21] = in[12][14];
out[2][23] = in[12][15];
out[14][0] = in[13][0];
out[10][2] = in[13][1];
out[14][3] = in[13][2];
out[10][5] = in[13][3];
out[14][6] = in[13][4];
out[10][8] = in[13][5];
out[14][9] = in[13][6];
out[10][11] = in[13][7];
out[14][12] = in[13][8];
out[10][14] = in[13][9];
out[14][15] = in[13][10];
out[10][17] = in[13][11];
out[14][18] = in[13][12];
out[10][20] = in[13][13];
out[14][21] = in[13][14];
out[10][23] = in[13][15];
out[7][0] = in[14][0];
out[3][2] = in[14][1];
out[7][3] = in[14][2];
out[3][5] = in[14][3];
out[7][6] = in[14][4];
out[3][8] = in[14][5];
out[7][9] = in[14][6];
out[3][11] = in[14][7];
out[7][12] = in[14][8];
out[3][14] = in[14][9];
out[7][15] = in[14][10];
out[3][17] = in[14][11];
out[7][18] = in[14][12];
out[3][20] = in[14][13];
out[7][21] = in[14][14];
out[3][23] = in[14][15];
out[15][0] = in[15][0];
out[11][2] = in[15][1];
out[15][3] = in[15][2];
out[11][5] = in[15][3];
out[15][6] = in[15][4];
out[11][8] = in[15][5];
out[15][9] = in[15][6];
out[11][11] = in[15][7];
out[15][12] = in[15][8];
out[11][14] = in[15][9];
out[15][15] = in[15][10];
out[11][17] = in[15][11];
out[15][18] = in[15][12];
out[11][20] = in[15][13];
out[15][21] = in[15][14];
out[11][23] = in[15][15];
out[0][1] = in[16][0];
out[4][2] = in[16][1];
out[0][4] = in[16][2];
out[4][5] = in[16][3];
out[0][7] = in[16][4];
out[4][8] = in[16][5];
out[0][10] = in[16][6];
out[4][11] = in[16][7];
out[0][13] = in[16][8];
out[4][14] = in[16][9];
out[0][16] = in[16][10];
out[4][17] = in[16][11];
out[0][19] = in[16][12];
out[4][20] = in[16][13];
out[0][22] = in[16][14];
out[4][23] = in[16][15];
out[8][1] = in[17][0];
out[12][2] = in[17][1];
out[8][4] = in[17][2];
out[12][5] = in[17][3];
out[8][7] = in[17][4];
out[12][8] = in[17][5];
out[8][10] = in[17][6];
out[12][11] = in[17][7];
out[8][13] = in[17][8];
out[12][14] = in[17][9];
out[8][16] = in[17][10];
out[12][17] = in[17][11];
out[8][19] = in[17][12];
out[12][20] = in[17][13];
out[8][22] = in[17][14];
out[12][23] = in[17][15];
out[1][1] = in[18][0];
out[5][2] = in[18][1];
out[1][4] = in[18][2];
out[5][5] = in[18][3];
out[1][7] = in[18][4];
out[5][8] = in[18][5];
out[1][10] = in[18][6];
out[5][11] = in[18][7];
out[1][13] = in[18][8];
out[5][14] = in[18][9];
out[1][16] = in[18][10];
out[5][17] = in[18][11];
out[1][19] = in[18][12];
out[5][20] = in[18][13];
out[1][22] = in[18][14];
out[5][23] = in[18][15];
out[9][1] = in[19][0];
out[13][2] = in[19][1];
out[9][4] = in[19][2];
out[13][5] = in[19][3];
out[9][7] = in[19][4];
out[13][8] = in[19][5];
out[9][10] = in[19][6];
out[13][11] = in[19][7];
out[9][13] = in[19][8];
out[13][14] = in[19][9];
out[9][16] = in[19][10];
out[13][17] = in[19][11];
out[9][19] = in[19][12];
out[13][20] = in[19][13];
out[9][22] = in[19][14];
out[13][23] = in[19][15];
out[2][1] = in[20][0];
out[6][2] = in[20][1];
out[2][4] = in[20][2];
out[6][5] = in[20][3];
out[2][7] = in[20][4];
out[6][8] = in[20][5];
out[2][10] = in[20][6];
out[6][11] = in[20][7];
out[2][13] = in[20][8];
out[6][14] = in[20][9];
out[2][16] = in[20][10];
out[6][17] = in[20][11];
out[2][19] = in[20][12];
out[6][20] = in[20][13];
out[2][22] = in[20][14];
out[6][23] = in[20][15];
out[10][1] = in[21][0];
out[14][2] = in[21][1];
out[10][4] = in[21][2];
out[14][5] = in[21][3];
out[10][7] = in[21][4];
out[14][8] = in[21][5];
out[10][10] = in[21][6];
out[14][11] = in[21][7];
out[10][13] = in[21][8];
out[14][14] = in[21][9];
out[10][16] = in[21][10];
out[14][17] = in[21][11];
out[10][19] = in[21][12];
out[14][20] = in[21][13];
out[10][22] = in[21][14];
out[14][23] = in[21][15];
out[3][1] = in[22][0];
out[7][2] = in[22][1];
out[3][4] = in[22][2];
out[7][5] = in[22][3];
out[3][7] = in[22][4];
out[7][8] = in[22][5];
out[3][10] = in[22][6];
out[7][11] = in[22][7];
out[3][13] = in[22][8];
out[7][14] = in[22][9];
out[3][16] = in[22][10];
out[7][17] = in[22][11];
out[3][19] = in[22][12];
out[7][20] = in[22][13];
out[3][22] = in[22][14];
out[7][23] = in[22][15];
out[11][1] = in[23][0];
out[15][2] = in[23][1];
out[11][4] = in[23][2];
out[15][5] = in[23][3];
out[11][7] = in[23][4];
out[15][8] = in[23][5];
out[11][10] = in[23][6];
out[15][11] = in[23][7];
out[11][13] = in[23][8];
out[15][14] = in[23][9];
out[11][16] = in[23][10];
out[15][17] = in[23][11];
out[11][19] = in[23][12];
out[15][20] = in[23][13];
out[11][22] = in[23][14];
out[15][23] = in[23][15];
}


// static void printBytes(const uint8_t* in, int len){
//     int i;
//     for(i=0;i<len;i++){
//         printf("%.2X", in[i]);
//     }
//     printf("\n");
// }

// void print_128(__m128i x){
//     uint32_t c[4];
//     _mm_storeu_si128((__m128i*)c+0,x);
//     // printf("3 = %8.8x\t  2 = %8.8x\t  1 = %8.8x\t  0 = %8.8x\t\n", c[3],c[2],c[1],c[0]);
//     printf("%8.8x %8.8x %8.8x %8.8x\n", c[0],c[1],c[2],c[3]);
// }

// void print_256(__m256i x){
//     uint64_t c[4];
//     _mm256_storeu_si256((__m256i*)c+0,x);
//     // printf("3 = %8.8x%8.8x\t  0 = %8.8x%8.8x\t\n", c[3],c[2],c[1],c[0]);
//     printf("%8.16lx%8.16lx %8.16lx%8.16lx\n", c[3],c[2],c[1],c[0]);
// }


uint8_t parity128(__m128i* s, uint8_t last){
    uint8_t p[16];
    for(size_t i = 1;i<8;i++){
        s[0] = _mm_xor_si128(s[0],s[i]);
    }
    
    _mm_storeu_si128((__m128i*)p,s[0]);
    for(size_t i = 0; i<16;i++){
        last = last ^ p[i];
    }
    return last;    
}


void matrix_mulBitslice(
    __m256i *output,
    __m256i *rIn,
    const uint8_t* matrix)
{
    __m256i temp, resp;
    __m256i aux[16], r[12], a;

    int i,j,c;

    for(i=0;i<12;i++){
	    r[i] = rIn[i];
    }
    
    i = 0;
    c = 0;
    
    while(1){
        
        loadLinha(i)/*linha 0*/
        a = resp;        
        i+=12;

        loadLinha(i)/* linha 9*/
        i+=12;
        aux[1] = resp;

        aux[0] = _mm256_unpacklo_epi8(a,aux[1]);/*b14a14b12a12b10a10b8a8b6a6b4a4b2a2b0a0*/
        aux[1] = _mm256_unpackhi_epi8(a,aux[1]);/*b15a15b13a13b11a11b9a9b7a7b5a5b3a3b1a1*/

        loadLinha(i)/*linha 18*/
        i+=12;
        a = resp;
        
        loadLinha(i)/*linha 27*/
        i+=12;
        aux[3] = resp;

        aux[2] = _mm256_unpacklo_epi8(a,aux[3]);/*d14c14d12c12d10c10d8c8d6c6d4c4d2c2d0c0*/
        aux[3] = _mm256_unpackhi_epi8(a,aux[3]);/*d15c15d13c13d11c11d9c9d7c7d5c5d3c3d1c1*/


        loadLinha(i)/*linha 36*/ /*leio e*/
        i+=12;
        a = resp;
        
        loadLinha(i)/*linha 45*/ /*leio f*/
        i+=12;
        aux[5] = resp;

        aux[4] = _mm256_unpacklo_epi8(a,aux[5]);/*f14e14f12e12f10e10f8e8f6e6f4e4f2e2f0e0*/
        aux[5] = _mm256_unpackhi_epi8(a,aux[5]);/*f15e15f13e13f11e11f9e9f7e7f5e5f3e3f1e1*/

        loadLinha(i)/*linha 54*/ /*leio g*/
        i+=12;
        a = resp;
        
        loadLinha(i)/*linha 63*/ /*leio h*/
        i+=12;
        aux[7] = resp;
        aux[6] = _mm256_unpacklo_epi8(a,aux[7]);/*h14g14h12g12h10g10h8g8h6g6h4g4h2g2h0g0*/
        aux[7] = _mm256_unpackhi_epi8(a,aux[7]);/*h15g15h13g13h11g11h9g9h7g7h5g5h3g3h1g1*/

        loadLinha(i)/*linha 72*/ /*leio i*/
        i+=12;
        a = resp;
        
        loadLinha(i)/*linha 81*/ /*leio j*/
        i+=12;
        aux[9] = resp;
        aux[8] = _mm256_unpacklo_epi8(a,aux[9]);/*j14i14j12i12j10i10j8i8j6i6j4i4j2i2j0i0*/
        aux[9] = _mm256_unpackhi_epi8(a,aux[9]);/*j15i15j13i13j11i11j9i9j7i7j5i5j3i3j1i1*/

        loadLinha(i)/*linha 90*/ /*leio k*/
        i+=12;
        a = resp;
        
        loadLinha(i)/*linha 99*/ /*leio l*/
        i+=12;
        aux[11] = resp;
        aux[10] = _mm256_unpacklo_epi8(a,aux[11]);/*l14k14l12k12l10k10l8k8l6k6l4k4l2k2l0k0*/
        aux[11] = _mm256_unpackhi_epi8(a,aux[11]);/*l15k15l13k13l11k11l9k9l7k7l5k5l3k3l1k1*/

        loadLinha(i)/*linha 108*/ /*leio m*/
        i+=12;
        a = resp;
        
        loadLinha(i)/*linha 117*/ /*leio n*/
        i+=12;
        aux[13] = resp;
        aux[12] = _mm256_unpacklo_epi8(a,aux[13]);/*n14m14n12m12n10m10n8m8n6m6n4m4n2m2n0m0*/
        aux[13] = _mm256_unpackhi_epi8(a,aux[13]);/*n15m15n13m13n11m11n9m9n7m7n5m5n3m3n1m1*/

        loadLinha(i)/*linha 90*/ /*leio k*/
        i+=12;
        a = resp;
        
        loadLinha(i)/*linha 126*/ /*leio o*/
        i+=12;
        aux[15] = resp;
        aux[14] = _mm256_unpacklo_epi8(a,aux[15]);/*p14o14p12o12p10o10p8o8p6o6p4o4p2o2p0o0*/
        aux[15] = _mm256_unpackhi_epi8(a,aux[15]);/*p15o15p13o13p11o11p9o9p7o7p5o5p3o3p1o1*/
        

/*-------------------------------------------------------------*/
        
        /*org*/
        a = _mm256_unpacklo_epi16(aux[0],aux[2]);/*d12c12b12a12d8c8b8a8d4c4b4a4d0c0b0a0*/
        aux[2] = _mm256_unpackhi_epi16(aux[0],aux[2]);/*d14c14b14a14d10c10b10a10d6c6b6a6d2c2b2a2*/
        aux[0] = a;

        a= _mm256_unpacklo_epi16(aux[1],aux[3]);/*d13c13b13a13d9c9b9a9d5c5b5a5d1c1b1a1*/
        aux[3] = _mm256_unpackhi_epi16(aux[1],aux[3]);/*d15c15b15a15d11c11b11a11d7c7b7a7d3c3b3a3*/
        aux[1] = a;

        a= _mm256_unpacklo_epi16(aux[4],aux[6]);/*h12g12f12e12h8g8f8e8h4g4f4e4h0g0f0e0*/
        aux[6] = _mm256_unpackhi_epi16(aux[4],aux[6]);/*h14g14f14e14h10g10f10e10h6g6f6e6h2g2f2e2*/
        aux[4] = a;

        a= _mm256_unpacklo_epi16(aux[5],aux[7]);/*h13g13f13e13h9g9f9e9h5g5f5e5h1g1f1e1*/
        aux[7] = _mm256_unpackhi_epi16(aux[5],aux[7]);/*h15g15f15e15h11g11f11e11h7g7f7e7h3g3f3e3*/
        aux[5] = a;

        a = _mm256_unpacklo_epi16(aux[8],aux[10]);/*l12k12j12i12l8k8j8i8l4k4j4i4l0k0j0i0*/
        aux[10] = _mm256_unpackhi_epi16(aux[8],aux[10]);/*l14k14j14i14l10k10j10i10l6k6j6i6l2k2j2i2*/
        aux[8] = a;

        a = _mm256_unpacklo_epi16(aux[9],aux[11]);/*l13k13j13i13l9k9j9i9l5k5j5i5l1k1j1i1*/
        aux[11] = _mm256_unpackhi_epi16(aux[9],aux[11]);/*l15k15j15i15l11k11j11i11l7k7j7i7l3k3j3i3*/
        aux[9] = a;

        a = _mm256_unpacklo_epi16(aux[12],aux[14]);/*p12o12n12m12p8o8n8m8p4o4n4m4p0o0n0m0*/
        aux[14] = _mm256_unpackhi_epi16(aux[12],aux[14]);/*p14o14n14m14p10o10n10m10p6o6n6m6p2o2n2m2*/
        aux[12] = a;

        a = _mm256_unpacklo_epi16(aux[13],aux[15]);/*p13o13n13m13p9o9n9m9p5o5n5m5p1o1n1m1*/
        aux[15] = _mm256_unpackhi_epi16(aux[13],aux[15]);/*p15o15n15m15p11o11n11m11p7o7n7m7p3o3n3m3*/
        aux[13] = a;

/*------------------------------------------------------------------------------------------------------------*/
        
        a = _mm256_unpacklo_epi32(aux[0],aux[4]);/*h08g08f08e08d08c08b08a08h00g00f00e00d00c00b00a00*/
        aux[4] = _mm256_unpackhi_epi32(aux[0],aux[4]);/*h12g12f12e12d12c12b12a12h04g04f04e04d04c04b04a04*/
        aux[0] = a;

        a = _mm256_unpacklo_epi32(aux[1],aux[5]);/*h09g09f09e09d09c09b09a09h01g01f01e01d01c01b01a01*/
        aux[5] = _mm256_unpackhi_epi32(aux[1],aux[5]);/*h13g13f13e13d13c13b13a13h05g05f05e05d05c05b05a05*/
        aux[1] = a;

        a = _mm256_unpacklo_epi32(aux[2],aux[6]);/*h10g10f10e10d10c10b10a10h02g02f02e02d02c02b02a02*/
        aux[6] = _mm256_unpackhi_epi32(aux[2],aux[6]);/*h14g14f14e14d14c14b14a14h06g06f06e06d06c06b06a06*/
        aux[2] = a;

        a = _mm256_unpacklo_epi32(aux[3],aux[7]);/*h11g11f11e11d11c11b11a11h03g03f03e03d03c03b03a03*/
        aux[7] = _mm256_unpackhi_epi32(aux[3],aux[7]);/*h15g15f15e15d15c15b15a11h07g07f07e07d07c07b07a07*/
        aux[3] = a;

        a = _mm256_unpacklo_epi32(aux[8],aux[12]);/*p08o08n08m08l08k08j08i08p00o00n00m00l00k00j00i00*/
        aux[12] = _mm256_unpackhi_epi32(aux[8],aux[12]);/*p12o12n12m12l12k12j12i12p04o04n04m04l04k04j04i04*/
        aux[8] = a;

        a = _mm256_unpacklo_epi32(aux[9],aux[13]);/*p09o09n09m09l09k09j09i09p01o01n01m01l01k01j01i01*/
        aux[13] = _mm256_unpackhi_epi32(aux[9],aux[13]);/*p13o13n13m13l13k13j13i13p05o05n05m05l05k05j05i05*/
        aux[9] = a;

        a = _mm256_unpacklo_epi32(aux[10],aux[14]);/*p10o10n10m10l10k10j10i10p02o02n02m02l02k02j02i02*/
        aux[14] = _mm256_unpackhi_epi32(aux[10],aux[14]);/*p14o14n14m14l14k14j14i14p06o06n06m06l06k06j06i06*/
        aux[10] = a;

        a = _mm256_unpacklo_epi32(aux[11],aux[15]);/*p11o11n11m11l11k11j11i11p03o03n03m03l03k03j03i03*/
        aux[15] = _mm256_unpackhi_epi32(aux[11],aux[15]);/*p15o15n15m15l15k15j15i11p07o07n07m07l07k07j07i07*/
        aux[11] = a;

/*------------------------------------------------------------------------------------------------------------*/
        a = _mm256_unpacklo_epi64(aux[0],aux[8]);/*p00o00n00m00l00k00j00i00h00g00f00e00d00c00b00a00*/
        aux[8] = _mm256_unpackhi_epi64(aux[0],aux[8]);/*p08o08n08m08l08k08j08i08h08g08f08e08d08c08b08a08*/
        aux[0] = a;

        a = _mm256_unpacklo_epi64(aux[1],aux[9]);/*p01o01n01m01l01k01j01i01h01g01f01e01d01c01b01a01*/
        aux[9] = _mm256_unpackhi_epi64(aux[1],aux[9]);/*p09o09n09m09l09k09j09i09h09g09f09e09d09c09b09a09*/
        aux[1] = a;

        a = _mm256_unpacklo_epi64(aux[2],aux[10]);/*p02o02n02m02l02k02j02i02h02g02f02e02d02c02b02a02*/
        aux[10] = _mm256_unpackhi_epi64(aux[2],aux[10]);/*p10o10n10m10l10k10j10i10h10g10f10e10d10c10b10a10*/
        aux[2] = a;
        
        a = _mm256_unpacklo_epi64(aux[3],aux[11]);/*p03o03n03m03l03k03j03i03h03g03f03e03d03c03b03a03*/
        aux[11] = _mm256_unpackhi_epi64(aux[3],aux[11]);/*p11o11n11m11l11k11j11i11h11g11f11e11d11c11b11a11*/
        aux[3] = a;

        a = _mm256_unpacklo_epi64(aux[4],aux[12]);/*p04o04n04m04l04k04j04i04h04g04f04e04d04c04b04a04*/
        aux[12] = _mm256_unpackhi_epi64(aux[4],aux[12]);/*p12o12n12m12l12k12j12i12h12g12f12e12d12c12b12a12*/
        aux[4] = a;

        a = _mm256_unpacklo_epi64(aux[5],aux[13]);/*p05o05n05m05l05k05j05i05h05g05f05e05d05c05b05a05*/
        aux[13] = _mm256_unpackhi_epi64(aux[5],aux[13]);/*p13o13n13m13l13k13j13i13h13g13f13e13d13c13b13a13*/
        aux[5] = a;

        a = _mm256_unpacklo_epi64(aux[6],aux[14]);/*p06o06n06m06l06k06j06i06h06g06f06e06d06c06b06a06*/
        aux[14] = _mm256_unpackhi_epi64(aux[6],aux[14]);/*p14o14n14m14l14k14j14i14h14g14f14e14d14c14b14a14*/
        aux[6] = a;

        a = _mm256_unpacklo_epi64(aux[7],aux[15]);/*p07o07n07m07l07k07j07i07h07g07f07e07d07c07b07a07*/
        aux[15] = _mm256_unpackhi_epi64(aux[7],aux[15]);/*p15o15n15m15l15k15j15i11h15g15f15e15d15c15b15a11*/
        aux[7] = a;

        output[c] = _mm256_setzero_si256();
        for(j=0;j<16;j++)
            output[c] = _mm256_xor_si256(output[c],aux[j]);

        if(i==203){
            break;
        }else if(i > 191){
            c++;
            i = c;
        }
    }
}

static void substitution(__m256i* r) {
    __m256i r1,r2,a,b,c;


    c = r[0];
    b = r[1];
    a = r[2];
    r1 = _mm256_xor_si256(a,_mm256_and_si256(b,c));
    r2 = _mm256_xor_si256(_mm256_xor_si256(a,b), _mm256_and_si256(a,c));
    r[0] = _mm256_xor_si256(_mm256_xor_si256(a,b), _mm256_xor_si256(c, _mm256_and_si256(a,b)));
    r[1] = r2;
    r[2] = r1;

    c = r[3];
    b = r[4];
    a = r[5];
    r1 = _mm256_xor_si256(a,_mm256_and_si256(b,c));
    r2 = _mm256_xor_si256(_mm256_xor_si256(a,b), _mm256_and_si256(a,c));
    r[3] = _mm256_xor_si256(_mm256_xor_si256(a,b), _mm256_xor_si256(c, _mm256_and_si256(a,b)));
    r[4] = r2;
    r[5] = r1;

    c = r[6];
    b = r[7];
    a = r[8];
    r1 = _mm256_xor_si256(a,_mm256_and_si256(b,c));
    r2 = _mm256_xor_si256(_mm256_xor_si256(a,b), _mm256_and_si256(a,c));
    r[6] = _mm256_xor_si256(_mm256_xor_si256(a,b), _mm256_xor_si256(c, _mm256_and_si256(a,b)));
    r[7] = r2;
    r[8] = r1;    
             
    c = r[9];
    b = r[10];
    a = r[11];
    r1 = _mm256_xor_si256(a,_mm256_and_si256(b,c));
    r2 = _mm256_xor_si256(_mm256_xor_si256(a,b), _mm256_and_si256(a,c));
    r[9] = _mm256_xor_si256(_mm256_xor_si256(a,b), _mm256_xor_si256(c, _mm256_and_si256(a,b)));
    r[10] = r2;
    r[11] = r1;

}




void LowMCEnc(uint8_t in[][16], uint8_t out[][16], uint8_t key[][16])
{
    int i;
    __m256i r[12],r2;
    __m256i rKey[12];
    __m256i rOut[12];
    __m256i rRoundKey[12];


    if (in != out) {
        for(i=0;i<12;i++)
	        rOut[i] =_mm256_loadu_si256((__m256i*)in+i);
    }

    for(i=0;i<12;i++){
	    r[i] =_mm256_loadu_si256((__m256i*)key+i);
        rKey[i] =_mm256_loadu_si256((__m256i*)key+i);
    }

    matrix_mulBitslice(rRoundKey, r, KMatrixBitslice(0, 192));      

    for(i = 0;i<12;i++)
        rOut[i] = _mm256_xor_si256(rOut[i],rRoundKey[i]);

   for (uint32_t r = 1; r <= ROUNDS; r++) {
        matrix_mulBitslice(rRoundKey, rKey, KMatrixBitslice(r, 192));
        substitution(rOut);        
        matrix_mulBitslice(rOut, rOut, LMatrixBitslice(r-1, 192)); 

        for(i = 0;i<12;i++){
            r2 = _mm256_loadu_si256((__m256i*)ConstantBitslice(r-1,192) + (i));
            rOut[i] = _mm256_xor_si256(rOut[i],r2);
        }

         for(i = 0;i<12;i++)
            rOut[i] = _mm256_xor_si256(rOut[i],rRoundKey[i]);    
    }
    for(i=0;i<12;i++){
	    _mm256_storeu_si256((__m256i*)out + i,rOut[i]);
    }
    
}


void LowMCEnc2(uint8_t in_t[][24], uint8_t out_t[][24], uint8_t key_t[][24])
{
    int i;
    __m256i r[12],r2;
    __m256i rKey[12];
    __m256i rOut[12];
    __m256i rRoundKey[12];

/************************************/
    ALIGN uint8_t aux[16][24] = {0};
    uint8_t in[24][16] = {0};
    uint8_t key[24][16] = {0};
    uint8_t out[24][16] = {0};

    stateTrans(&in_t[0],&aux[0]);
    stateTrans(&in_t[8],&aux[8]);

    stateTrans2(&aux[0],&in[0]);

    stateTrans(&key_t[0],&aux[0]);
    stateTrans(&key_t[8],&aux[8]);

    stateTrans2(&aux[0],&key[0]);
/************************************/

    if (in != out) {
        for(i=0;i<12;i++)
	        rOut[i] =_mm256_loadu_si256((__m256i*)in+i);
    }

    for(i=0;i<12;i++){
	    r[i] =_mm256_loadu_si256((__m256i*)key+i);
        rKey[i] =_mm256_loadu_si256((__m256i*)key+i);
    }

    matrix_mulBitslice(rRoundKey, r, KMatrixBitslice(0, 192));      

    for(i = 0;i<12;i++)
        rOut[i] = _mm256_xor_si256(rOut[i],rRoundKey[i]);

   for (uint32_t r = 1; r <= ROUNDS; r++) {
        matrix_mulBitslice(rRoundKey, rKey, KMatrixBitslice(r, 192));
        substitution(rOut);        
        matrix_mulBitslice(rOut, rOut, LMatrixBitslice(r-1, 192)); 

        for(i = 0;i<12;i++){
            r2 = _mm256_loadu_si256((__m256i*)ConstantBitslice(r-1,192) + (i));
            rOut[i] = _mm256_xor_si256(rOut[i],r2);
        }

         for(i = 0;i<12;i++)
            rOut[i] = _mm256_xor_si256(rOut[i],rRoundKey[i]);    
    }
    for(i=0;i<12;i++){
	    _mm256_storeu_si256((__m256i*)out + i,rOut[i]);
    }
    /*************************/
    

    stateTrans2Back(&out[0], &aux[0]);

    stateTransBack(&aux[0] , &out_t[0]);
    stateTransBack(&aux[8] , &out_t[8]);
    /*************************/    
}



int main(){
      uint8_t key[16][24] = {\
      {0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
       0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,},
      {0x81, 0xb8, 0x5d, 0xfe, 0x40, 0xf6, 0x12, 0x27, 0x5a, 0xa3, 0xf9, 0x19,
       0x91, 0x39, 0xeb, 0xaa, 0xe8, 0xdf, 0xf8, 0x36, 0x6f, 0x2d, 0xd3, 0x4e,},
      {0x24, 0x05, 0x97, 0x8f, 0xda, 0xad, 0x9b, 0x6d, 0x8d, 0xcd, 0xd1, 0x8a,
       0x0c, 0x2c, 0x0e, 0xc6, 0x8b, 0x69, 0xdd, 0x0a, 0x37, 0x54, 0xfe, 0x38,},
      {0x56, 0x9d, 0x7d, 0x82, 0x23, 0x00, 0x94, 0x3d, 0x94, 0x83, 0x47, 0x74,
       0x27, 0xe8, 0x8e, 0xa2, 0x27, 0xa2, 0xe3, 0x17, 0x2c, 0x04, 0xbc, 0xd3,},
       {0x81, 0xb8, 0x5d, 0xfe, 0x40, 0xf6, 0x12, 0x27, 0x5a, 0xa3, 0xf9, 0x19,
       0x91, 0x39, 0xeb, 0xaa, 0xe8, 0xdf, 0xf8, 0x36, 0x6f, 0x2d, 0xd3, 0x4e,},
      {0x24, 0x05, 0x97, 0x8f, 0xda, 0xad, 0x9b, 0x6d, 0x8d, 0xcd, 0xd1, 0x8a,
       0x0c, 0x2c, 0x0e, 0xc6, 0x8b, 0x69, 0xdd, 0x0a, 0x37, 0x54, 0xfe, 0x38,},
      {0x56, 0x9d, 0x7d, 0x82, 0x23, 0x00, 0x94, 0x3d, 0x94, 0x83, 0x47, 0x74,
       0x27, 0xe8, 0x8e, 0xa2, 0x27, 0xa2, 0xe3, 0x17, 0x2c, 0x04, 0xbc, 0xd3,},
       {0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
       0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,},
       {0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
       0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,},
      {0x81, 0xb8, 0x5d, 0xfe, 0x40, 0xf6, 0x12, 0x27, 0x5a, 0xa3, 0xf9, 0x19,
       0x91, 0x39, 0xeb, 0xaa, 0xe8, 0xdf, 0xf8, 0x36, 0x6f, 0x2d, 0xd3, 0x4e,},
      {0x24, 0x05, 0x97, 0x8f, 0xda, 0xad, 0x9b, 0x6d, 0x8d, 0xcd, 0xd1, 0x8a,
       0x0c, 0x2c, 0x0e, 0xc6, 0x8b, 0x69, 0xdd, 0x0a, 0x37, 0x54, 0xfe, 0x38,},
      {0x56, 0x9d, 0x7d, 0x82, 0x23, 0x00, 0x94, 0x3d, 0x94, 0x83, 0x47, 0x74,
       0x27, 0xe8, 0x8e, 0xa2, 0x27, 0xa2, 0xe3, 0x17, 0x2c, 0x04, 0xbc, 0xd3,},
       {0x81, 0xb8, 0x5d, 0xfe, 0x40, 0xf6, 0x12, 0x27, 0x5a, 0xa3, 0xf9, 0x19,
       0x91, 0x39, 0xeb, 0xaa, 0xe8, 0xdf, 0xf8, 0x36, 0x6f, 0x2d, 0xd3, 0x4e,},
      {0x24, 0x05, 0x97, 0x8f, 0xda, 0xad, 0x9b, 0x6d, 0x8d, 0xcd, 0xd1, 0x8a,
       0x0c, 0x2c, 0x0e, 0xc6, 0x8b, 0x69, 0xdd, 0x0a, 0x37, 0x54, 0xfe, 0x38,},
      {0x56, 0x9d, 0x7d, 0x82, 0x23, 0x00, 0x94, 0x3d, 0x94, 0x83, 0x47, 0x74,
       0x27, 0xe8, 0x8e, 0xa2, 0x27, 0xa2, 0xe3, 0x17, 0x2c, 0x04, 0xbc, 0xd3,},
       {0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 
       0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,}
  };

  uint8_t plaintext[16][24] = {
      {0xab, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
       0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,},
      {0xb8, 0x65, 0xcc, 0xf3, 0xfc, 0xda, 0x8d, 0xdb, 0xed, 0x52, 0x7d, 0xc3,
       0x4d, 0xd4, 0x15, 0xd,  0x4a, 0x48, 0x2d, 0xcb, 0xf7, 0xe9, 0x64, 0x3c,},
      {0x33, 0xe8, 0xb4, 0x55, 0x2e, 0x95, 0xef, 0x52, 0x79, 0x49, 0x77, 0x06,
       0xbc, 0xe0, 0x1e, 0xcb, 0x4a, 0xcb, 0x86, 0x01, 0x41, 0xb7, 0xfc, 0x43,},
      {0xae, 0xeb, 0x9d, 0x5b, 0x61, 0xa2, 0xa5, 0x6d, 0xd5, 0x98, 0xf7, 0xda,
       0x26, 0xdf, 0xd7, 0x8c, 0xc9, 0x92, 0xe0, 0xae, 0xa3, 0xfc, 0x2e, 0x39,},
      {0xb8, 0x65, 0xcc, 0xf3, 0xfc, 0xda, 0x8d, 0xdb, 0xed, 0x52, 0x7d, 0xc3,
       0x4d, 0xd4, 0x15, 0xd,  0x4a, 0x48, 0x2d, 0xcb, 0xf7, 0xe9, 0x64, 0x3c,},
      {0x33, 0xe8, 0xb4, 0x55, 0x2e, 0x95, 0xef, 0x52, 0x79, 0x49, 0x77, 0x06,
       0xbc, 0xe0, 0x1e, 0xcb, 0x4a, 0xcb, 0x86, 0x01, 0x41, 0xb7, 0xfc, 0x43,},
      {0xae, 0xeb, 0x9d, 0x5b, 0x61, 0xa2, 0xa5, 0x6d, 0xd5, 0x98, 0xf7, 0xda,
       0x26, 0xdf, 0xd7, 0x8c, 0xc9, 0x92, 0xe0, 0xae, 0xa3, 0xfc, 0x2e, 0x39,},
       {0xab, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
       0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,},
       {0xab, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
       0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,},
      {0xb8, 0x65, 0xcc, 0xf3, 0xfc, 0xda, 0x8d, 0xdb, 0xed, 0x52, 0x7d, 0xc3,
       0x4d, 0xd4, 0x15, 0xd,  0x4a, 0x48, 0x2d, 0xcb, 0xf7, 0xe9, 0x64, 0x3c,},
      {0x33, 0xe8, 0xb4, 0x55, 0x2e, 0x95, 0xef, 0x52, 0x79, 0x49, 0x77, 0x06,
       0xbc, 0xe0, 0x1e, 0xcb, 0x4a, 0xcb, 0x86, 0x01, 0x41, 0xb7, 0xfc, 0x43,},
      {0xae, 0xeb, 0x9d, 0x5b, 0x61, 0xa2, 0xa5, 0x6d, 0xd5, 0x98, 0xf7, 0xda,
       0x26, 0xdf, 0xd7, 0x8c, 0xc9, 0x92, 0xe0, 0xae, 0xa3, 0xfc, 0x2e, 0x39,},
      {0xb8, 0x65, 0xcc, 0xf3, 0xfc, 0xda, 0x8d, 0xdb, 0xed, 0x52, 0x7d, 0xc3,
       0x4d, 0xd4, 0x15, 0xd,  0x4a, 0x48, 0x2d, 0xcb, 0xf7, 0xe9, 0x64, 0x3c,},
      {0x33, 0xe8, 0xb4, 0x55, 0x2e, 0x95, 0xef, 0x52, 0x79, 0x49, 0x77, 0x06,
       0xbc, 0xe0, 0x1e, 0xcb, 0x4a, 0xcb, 0x86, 0x01, 0x41, 0xb7, 0xfc, 0x43,},
      {0xae, 0xeb, 0x9d, 0x5b, 0x61, 0xa2, 0xa5, 0x6d, 0xd5, 0x98, 0xf7, 0xda,
       0x26, 0xdf, 0xd7, 0x8c, 0xc9, 0x92, 0xe0, 0xae, 0xa3, 0xfc, 0x2e, 0x39,},
       {0xab, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
       0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,}
  };
  uint8_t ciphertext_expected[16][24] = {\
      {0xf8, 0xf7, 0xa2, 0x25, 0xde, 0x77, 0x12, 0x31, 0x29, 0x10, 0x7a, 0x20,
       0xf5, 0x54, 0x3a, 0xfa, 0x78, 0x33, 0x7,  0x66, 0x53, 0xba, 0x2b, 0x29},
      {0x95, 0xef, 0x9e, 0xd7, 0xc3, 0x78, 0x72, 0xa7, 0xb4, 0x60, 0x2a, 0x3f,
       0xa9, 0xc4, 0x6e, 0xbc, 0xb8, 0x42, 0x54, 0xed, 0xe,  0x44, 0xee, 0x9f,},
      {0xdd, 0xaf, 0xf,  0x9d, 0x9e, 0xdd, 0x57, 0x20, 0x69, 0xa8, 0x94, 0x9f,
       0xae, 0xa0, 0xd1, 0xfd, 0x2d, 0x91, 0xef, 0x26, 0x2b, 0x41, 0x1c, 0xaf},
      {0x86, 0x98, 0x70, 0xae, 0x65, 0x47, 0xad, 0x0a, 0xfe, 0xf2, 0x77, 0x93,
       0x17, 0x0d, 0x96, 0xbc, 0x78, 0xe0, 0x40, 0x09, 0x69, 0x44, 0x80, 0x8f},
        {0x95, 0xef, 0x9e, 0xd7, 0xc3, 0x78, 0x72, 0xa7, 0xb4, 0x60, 0x2a, 0x3f,
       0xa9, 0xc4, 0x6e, 0xbc, 0xb8, 0x42, 0x54, 0xed, 0xe,  0x44, 0xee, 0x9f,},
      {0xdd, 0xaf, 0xf,  0x9d, 0x9e, 0xdd, 0x57, 0x20, 0x69, 0xa8, 0x94, 0x9f,
       0xae, 0xa0, 0xd1, 0xfd, 0x2d, 0x91, 0xef, 0x26, 0x2b, 0x41, 0x1c, 0xaf},
      {0x86, 0x98, 0x70, 0xae, 0x65, 0x47, 0xad, 0x0a, 0xfe, 0xf2, 0x77, 0x93,
       0x17, 0x0d, 0x96, 0xbc, 0x78, 0xe0, 0x40, 0x09, 0x69, 0x44, 0x80, 0x8f},
       {0xf8, 0xf7, 0xa2, 0x25, 0xde, 0x77, 0x12, 0x31, 0x29, 0x10, 0x7a, 0x20,
       0xf5, 0x54, 0x3a, 0xfa, 0x78, 0x33, 0x7,  0x66, 0x53, 0xba, 0x2b, 0x29},
       {0xf8, 0xf7, 0xa2, 0x25, 0xde, 0x77, 0x12, 0x31, 0x29, 0x10, 0x7a, 0x20,
       0xf5, 0x54, 0x3a, 0xfa, 0x78, 0x33, 0x7,  0x66, 0x53, 0xba, 0x2b, 0x29},
      {0x95, 0xef, 0x9e, 0xd7, 0xc3, 0x78, 0x72, 0xa7, 0xb4, 0x60, 0x2a, 0x3f,
       0xa9, 0xc4, 0x6e, 0xbc, 0xb8, 0x42, 0x54, 0xed, 0xe,  0x44, 0xee, 0x9f,},
      {0xdd, 0xaf, 0xf,  0x9d, 0x9e, 0xdd, 0x57, 0x20, 0x69, 0xa8, 0x94, 0x9f,
       0xae, 0xa0, 0xd1, 0xfd, 0x2d, 0x91, 0xef, 0x26, 0x2b, 0x41, 0x1c, 0xaf},
      {0x86, 0x98, 0x70, 0xae, 0x65, 0x47, 0xad, 0x0a, 0xfe, 0xf2, 0x77, 0x93,
       0x17, 0x0d, 0x96, 0xbc, 0x78, 0xe0, 0x40, 0x09, 0x69, 0x44, 0x80, 0x8f},
        {0x95, 0xef, 0x9e, 0xd7, 0xc3, 0x78, 0x72, 0xa7, 0xb4, 0x60, 0x2a, 0x3f,
       0xa9, 0xc4, 0x6e, 0xbc, 0xb8, 0x42, 0x54, 0xed, 0xe,  0x44, 0xee, 0x9f,},
      {0xdd, 0xaf, 0xf,  0x9d, 0x9e, 0xdd, 0x57, 0x20, 0x69, 0xa8, 0x94, 0x9f,
       0xae, 0xa0, 0xd1, 0xfd, 0x2d, 0x91, 0xef, 0x26, 0x2b, 0x41, 0x1c, 0xaf},
      {0x86, 0x98, 0x70, 0xae, 0x65, 0x47, 0xad, 0x0a, 0xfe, 0xf2, 0x77, 0x93,
       0x17, 0x0d, 0x96, 0xbc, 0x78, 0xe0, 0x40, 0x09, 0x69, 0x44, 0x80, 0x8f},
       {0xf8, 0xf7, 0xa2, 0x25, 0xde, 0x77, 0x12, 0x31, 0x29, 0x10, 0x7a, 0x20,
       0xf5, 0x54, 0x3a, 0xfa, 0x78, 0x33, 0x7,  0x66, 0x53, 0xba, 0x2b, 0x29}
    };

    ALIGN uint8_t aux[16][24] = {0}; 
    ALIGN uint8_t aux2[16][24] = {0};
    ALIGN uint8_t out[16][24] = {0};


    uint8_t plaintextAux2[24][16] = {0};
    uint8_t keyAux2[24][16] = {0};
    uint8_t outAux[24][16] = {0};

    stateTrans(&plaintext[0],&aux[0]);
    stateTrans(&plaintext[8],&aux[8]);

    stateTrans2(aux,plaintextAux2);

    stateTrans(&key[0],&aux2[0]);
    stateTrans(&key[8],&aux2[8]);

    stateTrans2(&aux2[0],&keyAux2[0]);
    LowMCEnc(plaintextAux2, outAux, keyAux2);

    stateTrans2Back(&outAux[0],&aux[0]);
    stateTransBack(&aux[0] , &out[0]);
    stateTransBack(&aux[8] , &out[8]);


    int res = 0;
    for(int j=0;j<16;j++){
        for(int i=0;i<17;i++){
            if(out[j][i] != ciphertext_expected[j][i])
                res++;
        }
    }
    if(res){
        printf("Saída incorreta!!!\n\n");
        exit(1);
    }
    printf("Correto!!\n\n");
    BENCH_FUNCTION(LowMCEnc,plaintextAux2, outAux, keyAux2);


    printf("Completo!\n\n");
    
    LowMCEnc2(plaintext, out, key);

    res = 0;
    for(int j=0;j<16;j++){
        for(int i=0;i<17;i++){
            if(out[j][i] != ciphertext_expected[j][i])
                res++;
        }
    }
    if(res){
        printf("Saída incorreta!!!\n\n");
        exit(1);
    }
    printf("Correto!");
    BENCH_FUNCTION(LowMCEnc2,plaintext, out, key);

    return 0;        
}



