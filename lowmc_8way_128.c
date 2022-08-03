/*! @file lowmc_8way_128.c
 *  @brief 8-way optimized implementation of the LowMC-129-129-4   
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

#define loadLinha(i)            resp =  _mm_setzero_si128();\
                                for(j=0;j<9;j++){\
                                    temp = _mm_load_si128((__m128i*)matrix+(i*9+j));\
                                    temp = _mm_and_si128(temp,r[j]);\
                                    resp = _mm_xor_si128(resp,temp);\
                                }\

uint64_t transpose8(uint64_t x) {

    x = (x & 0xAA55AA55AA55AA55LL) | ((x & 0x00AA00AA00AA00AALL) << 7)  | ((x >> 7) & 0x00AA00AA00AA00AALL);
    x = (x & 0xCCCC3333CCCC3333LL) | ((x & 0x0000CCCC0000CCCCLL) << 14) | ((x >> 14) & 0x0000CCCC0000CCCCLL);
    x = (x & 0xF0F0F0F00F0F0F0FLL) | ((x & 0x00000000F0F0F0F0LL) << 28) | ((x >> 28) & 0x00000000F0F0F0F0LL);
    
    return x;

} 

int stateTransBack(uint8_t in[][17], uint8_t out[][17]){
    int i,j,k;
    uint64_t x;
    int t=0;
    for(i=0;i<16;i++){
        x = 0;
        for(j=0;j<8;j++){
            x = x << 8 | in[j][i];  
        }
        x = transpose8(x);
        for (k = t+7; k >= t; k--) { 
            out[k][i] = x; x = x >> 8;
        }
    }

    i = 7;
    for (k = t+7; k >= t; k--, i--) 
        out[k][16] = (in[0][16] << i) & 0x80;
    return t;
}

void stateTrans(uint8_t in[][17], uint8_t out[][17]){
    int j,k;
    uint8_t comp = 0;
    
    uint64_t x = 0;
    for(j=0;j<16;j++){
        for(k=0;k<8;k++){
            x = x << 8 | in[k][j];  
        }
        x = transpose8(x);
        for (k = 7; k >= 0; k--) { 
            out[k][j] = x; x = x >> 8;
        }
    } 
    
    comp = 0;
    j = 0;
    for(k=0;k<8;k++){
        comp = (comp) | ((in[k][16] & 0x80) >> j++);
    }
    out[0][16] = comp;
}

void stateTrans2(uint8_t in[][17], uint8_t out[][16]){
out[0][0] = in[0][0];
out[8][0] = in[0][1];
out[7][1] = in[0][2];
out[6][2] = in[0][3];
out[5][3] = in[0][4];
out[4][4] = in[0][5];
out[3][5] = in[0][6];
out[2][6] = in[0][7];
out[1][7] = in[0][8];
out[0][8] = in[0][9];
out[8][8] = in[0][10];
out[7][9] = in[0][11];
out[6][10] = in[0][12];
out[5][11] = in[0][13];
out[4][12] = in[0][14];
out[3][13] = in[0][15];
out[2][14] = in[0][16];


out[1][0] = in[1][0];
out[0][1] = in[1][1];
out[8][1] = in[1][2];
out[7][2] = in[1][3];
out[6][3] = in[1][4];
out[5][4] = in[1][5];
out[4][5] = in[1][6];
out[3][6] = in[1][7];
out[2][7] = in[1][8];
out[1][8] = in[1][9];
out[0][9] = in[1][10];
out[8][9] = in[1][11];
out[7][10]= in[1][12];
out[6][11]= in[1][13];
out[5][12]= in[1][14];
out[4][13]= in[1][15];


out[2][0] = in[2][0];
out[1][1] = in[2][1];
out[0][2] = in[2][2];
out[8][2] = in[2][3];
out[7][3] = in[2][4];
out[6][4] = in[2][5];
out[5][5] = in[2][6];
out[4][6] = in[2][7];
out[3][7] = in[2][8];
out[2][8] = in[2][9];
out[1][9] = in[2][10];
out[0][10]= in[2][11];
out[8][10]= in[2][12];
out[7][11]= in[2][13];
out[6][12]= in[2][14];
out[5][13]= in[2][15];

out[3][0] = in[3][0];
out[2][1] = in[3][1];
out[1][2] = in[3][2];
out[0][3] = in[3][3];
out[8][3] = in[3][4];
out[7][4] = in[3][5];
out[6][5] = in[3][6];
out[5][6] = in[3][7];
out[4][7] = in[3][8];
out[3][8] = in[3][9];
out[2][9] = in[3][10];
out[1][10]= in[3][11];
out[0][11]= in[3][12];
out[8][11]= in[3][13];
out[7][12]= in[3][14];
out[6][13]= in[3][15];


out[4][0] = in[4][0];
out[3][1] = in[4][1];
out[2][2] = in[4][2];
out[1][3] = in[4][3];
out[8][4] = in[4][5];
out[0][4] = in[4][4];
out[7][5] = in[4][6];
out[6][6] = in[4][7];
out[5][7] = in[4][8];
out[4][8] = in[4][9];
out[3][9] = in[4][10];
out[2][10]= in[4][11];
out[1][11]= in[4][12];
out[0][12]= in[4][13];
out[8][12]= in[4][14];
out[7][13]= in[4][15];

out[5][0] = in[5][0];
out[4][1] = in[5][1];
out[3][2] = in[5][2];
out[2][3] = in[5][3];
out[1][4] = in[5][4];
out[0][5] = in[5][5];
out[8][5] = in[5][6];
out[7][6] = in[5][7];
out[6][7] = in[5][8];
out[5][8] = in[5][9];
out[4][9] = in[5][10];
out[3][10]= in[5][11];
out[2][11]= in[5][12];
out[1][12]= in[5][13];
out[0][13]= in[5][14];
out[8][13]= in[5][15];


out[6][0] = in[6][0];
out[5][1] = in[6][1];
out[4][2] = in[6][2];
out[3][3] = in[6][3];
out[2][4] = in[6][4];
out[1][5] = in[6][5];
out[0][6] = in[6][6];
out[8][6] = in[6][7];
out[7][7] = in[6][8];
out[6][8] = in[6][9];
out[5][9] = in[6][10];
out[4][10]= in[6][11];
out[3][11]= in[6][12];
out[2][12]= in[6][13];
out[1][13]= in[6][14];
out[0][14]= in[6][15];

out[7][0] = in[7][0];
out[6][1] = in[7][1];
out[5][2] = in[7][2];
out[4][3] = in[7][3];
out[3][4] = in[7][4];
out[2][5] = in[7][5];
out[1][6] = in[7][6];
out[0][7] = in[7][7];
out[8][7] = in[7][8];
out[7][8] = in[7][9];
out[6][9] = in[7][10];
out[5][10]= in[7][11];
out[4][11]= in[7][12];
out[3][12]= in[7][13];
out[2][13]= in[7][14];
out[1][14]= in[7][15];

}

void stateTrans2Back(uint8_t in[][16], uint8_t out[][17]){
out[0][0] = in[0][0];
out[1][1] = in[0][1];
out[2][2] = in[0][2];
out[3][3] = in[0][3];
out[4][4] = in[0][4];
out[5][5] = in[0][5];
out[6][6] = in[0][6];
out[7][7] = in[0][7];
out[0][9] = in[0][8];
out[1][10] = in[0][9];
out[2][11] = in[0][10];
out[3][12] = in[0][11];
out[4][13] = in[0][12];
out[5][14] = in[0][13];
out[6][15] = in[0][14];

out[1][0] = in[1][0];
out[2][1] = in[1][1];
out[3][2] = in[1][2];
out[4][3] = in[1][3];
out[5][4] = in[1][4];
out[6][5] = in[1][5];
out[7][6] = in[1][6];
out[0][8] = in[1][7];
out[1][9] = in[1][8];
out[2][10] = in[1][9];
out[3][11] = in[1][10];
out[4][12] = in[1][11];
out[5][13] = in[1][12];
out[6][14] = in[1][13];
out[7][15] = in[1][14];

out[2][0] = in[2][0];
out[3][1] = in[2][1];
out[4][2] = in[2][2];
out[5][3] = in[2][3];
out[6][4] = in[2][4];
out[7][5] = in[2][5];
out[0][7] = in[2][6];
out[1][8] = in[2][7];
out[2][9] = in[2][8];
out[3][10] = in[2][9];
out[4][11] = in[2][10];
out[5][12] = in[2][11];
out[6][13] = in[2][12];
out[7][14] = in[2][13];
out[0][16] = in[2][14];

out[3][0] = in[3][0];
out[4][1] = in[3][1];
out[5][2] = in[3][2];
out[6][3] = in[3][3];
out[7][4] = in[3][4];
out[0][6] = in[3][5];
out[1][7] = in[3][6];
out[2][8] = in[3][7];
out[3][9] = in[3][8];
out[4][10] = in[3][9];
out[5][11] = in[3][10];
out[6][12] = in[3][11];
out[7][13] = in[3][12];
out[0][15] = in[3][13];

out[4][0] = in[4][0];
out[5][1] = in[4][1];
out[6][2] = in[4][2];
out[7][3] = in[4][3];
out[0][5] = in[4][4];
out[1][6] = in[4][5];
out[2][7] = in[4][6];
out[3][8] = in[4][7];
out[4][9] = in[4][8];
out[5][10] = in[4][9];
out[6][11] = in[4][10];
out[7][12] = in[4][11];
out[0][14] = in[4][12];
out[1][15] = in[4][13];

out[5][0] = in[5][0];
out[6][1] = in[5][1];
out[7][2] = in[5][2];
out[0][4] = in[5][3];
out[1][5] = in[5][4];
out[2][6] = in[5][5];
out[3][7] = in[5][6];
out[4][8] = in[5][7];
out[5][9] = in[5][8];
out[6][10] = in[5][9];
out[7][11] = in[5][10];
out[0][13] = in[5][11];
out[1][14] = in[5][12];
out[2][15] = in[5][13];

out[6][0] = in[6][0];
out[7][1] = in[6][1];
out[0][3] = in[6][2];
out[1][4] = in[6][3];
out[2][5] = in[6][4];
out[3][6] = in[6][5];
out[4][7] = in[6][6];
out[5][8] = in[6][7];
out[6][9] = in[6][8];
out[7][10] = in[6][9];
out[0][12] = in[6][10];
out[1][13] = in[6][11];
out[2][14] = in[6][12];
out[3][15] = in[6][13];
out[7][0] = in[7][0];
out[0][2] = in[7][1];
out[1][3] = in[7][2];
out[2][4] = in[7][3];
out[3][5] = in[7][4];
out[4][6] = in[7][5];
out[5][7] = in[7][6];
out[6][8] = in[7][7];
out[7][9] = in[7][8];
out[0][11] = in[7][9];
out[1][12] = in[7][10];
out[2][13] = in[7][11];
out[3][14] = in[7][12];
out[4][15] = in[7][13];
out[0][1] = in[8][0];
out[1][2] = in[8][1];
out[2][3] = in[8][2];
out[3][4] = in[8][3];
out[4][5] = in[8][4];
out[5][6] = in[8][5];
out[6][7] = in[8][6];
out[7][8] = in[8][7];
out[0][10] = in[8][8];
out[1][11] = in[8][9];
out[2][12] = in[8][10];
out[3][13] = in[8][11];
out[4][14] = in[8][12];
out[5][15] = in[8][13];

}

static void printBytes(const uint8_t* in, int len){
    int i;
    for(i=0;i<len;i++){
        printf("%.2X", in[i]);
    }
    printf("\n");
}

// void print_128(__m128i x){
//     uint32_t c[4];
//     _mm_storeu_si128((__m128i*)c+0,x);
//     // printf("3 = %8.8x\t  2 = %8.8x\t  1 = %8.8x\t  0 = %8.8x\t\n", c[3],c[2],c[1],c[0]);
//     printf("%8.8x%8.8x%8.8x%8.8x\n", c[3],c[2],c[1],c[0]);
// }




void matrix_mulBitslice(
    __m128i *output,
    __m128i *rIn,
    const uint8_t* matrix)
{
    __m128i temp, resp;
    __m128i aux[16], r[9], a;

    int i,j,c;

    for(i=0;i<9;i++){
	    r[i] = rIn[i];
    }
    
    i = 0;
    c = 0;
    
    while(1){
        loadLinha(i)
        a = resp;
        i+=9;
        loadLinha(i)/* linha 9*/
        i+=9;
        aux[1] = resp;

        aux[0] = _mm_unpacklo_epi8(a,aux[1]);/*b14a14b12a12b10a10b8a8b6a6b4a4b2a2b0a0*/
        aux[1] = _mm_unpackhi_epi8(a,aux[1]);/*b15a15b13a13b11a11b9a9b7a7b5a5b3a3b1a1*/

        loadLinha(i)/*linha 18*/
        i+=9;
        a = resp;
        
        loadLinha(i)/*linha 27*/
        i+=9;
        aux[3] = resp;

        aux[2] = _mm_unpacklo_epi8(a,aux[3]);/*d14c14d12c12d10c10d8c8d6c6d4c4d2c2d0c0*/
        aux[3] = _mm_unpackhi_epi8(a,aux[3]);/*d15c15d13c13d11c11d9c9d7c7d5c5d3c3d1c1*/


        loadLinha(i)/*linha 36*/ /*leio e*/
        i+=9;
        a = resp;
        
        loadLinha(i)/*linha 45*/ /*leio f*/
        i+=9;
        aux[5] = resp;

        aux[4] = _mm_unpacklo_epi8(a,aux[5]);/*f14e14f12e12f10e10f8e8f6e6f4e4f2e2f0e0*/
        aux[5] = _mm_unpackhi_epi8(a,aux[5]);/*f15e15f13e13f11e11f9e9f7e7f5e5f3e3f1e1*/

        loadLinha(i)/*linha 54*/ /*leio g*/
        i+=9;
        a = resp;
        
        loadLinha(i)/*linha 63*/ /*leio h*/
        i+=9;
        aux[7] = resp;
        aux[6] = _mm_unpacklo_epi8(a,aux[7]);/*h14g14h12g12h10g10h8g8h6g6h4g4h2g2h0g0*/
        aux[7] = _mm_unpackhi_epi8(a,aux[7]);/*h15g15h13g13h11g11h9g9h7g7h5g5h3g3h1g1*/

        loadLinha(i)/*linha 72*/ /*leio i*/
        i+=9;
        a = resp;
        
        loadLinha(i)/*linha 81*/ /*leio j*/
        i+=9;
        aux[9] = resp;
        aux[8] = _mm_unpacklo_epi8(a,aux[9]);/*j14i14j12i12j10i10j8i8j6i6j4i4j2i2j0i0*/
        aux[9] = _mm_unpackhi_epi8(a,aux[9]);/*j15i15j13i13j11i11j9i9j7i7j5i5j3i3j1i1*/

        loadLinha(i)/*linha 90*/ /*leio k*/
        i+=9;
        a = resp;
        
        loadLinha(i)/*linha 99*/ /*leio l*/
        i+=9;
        aux[11] = resp;
        aux[10] = _mm_unpacklo_epi8(a,aux[11]);/*l14k14l12k12l10k10l8k8l6k6l4k4l2k2l0k0*/
        aux[11] = _mm_unpackhi_epi8(a,aux[11]);/*l15k15l13k13l11k11l9k9l7k7l5k5l3k3l1k1*/

        loadLinha(i)/*linha 108*/ /*leio m*/
        i+=9;
        a = resp;
        
        loadLinha(i)/*linha 117*/ /*leio n*/
        i+=9;
        aux[13] = resp;
        aux[12] = _mm_unpacklo_epi8(a,aux[13]);/*n14m14n12m12n10m10n8m8n6m6n4m4n2m2n0m0*/
        aux[13] = _mm_unpackhi_epi8(a,aux[13]);/*n15m15n13m13n11m11n9m9n7m7n5m5n3m3n1m1*/

        if(c < 3){
            loadLinha(i)/*linha 126*/ /*leio o*/
            i+=9;
            a = resp;
            
            /*linha LIXO*/ /*leio p*/
            aux[14] = _mm_unpacklo_epi8(a,_mm_setzero_si128());/*p14o14p12o12p10o10p8o8p6o6p4o4p2o2p0o0*/
            aux[15] = _mm_unpackhi_epi8(a,_mm_setzero_si128());/*p15o15p13o13p11o11p9o9p7o7p5o5p3o3p1o1*/
        }else{
            i+=9;
            aux[14] = _mm_setzero_si128();
            aux[15] = _mm_setzero_si128();
        }

        
        /*org*/
        a = _mm_unpacklo_epi16(aux[0],aux[2]);/*d12c12b12a12d8c8b8a8d4c4b4a4d0c0b0a0*/
        aux[2] = _mm_unpackhi_epi16(aux[0],aux[2]);/*d14c14b14a14d10c10b10a10d6c6b6a6d2c2b2a2*/
        aux[0] = a;

        a= _mm_unpacklo_epi16(aux[1],aux[3]);/*d13c13b13a13d9c9b9a9d5c5b5a5d1c1b1a1*/
        aux[3] = _mm_unpackhi_epi16(aux[1],aux[3]);/*d15c15b15a15d11c11b11a11d7c7b7a7d3c3b3a3*/
        aux[1] = a;

        a= _mm_unpacklo_epi16(aux[4],aux[6]);/*h12g12f12e12h8g8f8e8h4g4f4e4h0g0f0e0*/
        aux[6] = _mm_unpackhi_epi16(aux[4],aux[6]);/*h14g14f14e14h10g10f10e10h6g6f6e6h2g2f2e2*/
        aux[4] = a;

        a= _mm_unpacklo_epi16(aux[5],aux[7]);/*h13g13f13e13h9g9f9e9h5g5f5e5h1g1f1e1*/
        aux[7] = _mm_unpackhi_epi16(aux[5],aux[7]);/*h15g15f15e15h11g11f11e11h7g7f7e7h3g3f3e3*/
        aux[5] = a;

        a = _mm_unpacklo_epi16(aux[8],aux[10]);/*l12k12j12i12l8k8j8i8l4k4j4i4l0k0j0i0*/
        aux[10] = _mm_unpackhi_epi16(aux[8],aux[10]);/*l14k14j14i14l10k10j10i10l6k6j6i6l2k2j2i2*/
        aux[8] = a;

        a = _mm_unpacklo_epi16(aux[9],aux[11]);/*l13k13j13i13l9k9j9i9l5k5j5i5l1k1j1i1*/
        aux[11] = _mm_unpackhi_epi16(aux[9],aux[11]);/*l15k15j15i15l11k11j11i11l7k7j7i7l3k3j3i3*/
        aux[9] = a;

        a = _mm_unpacklo_epi16(aux[12],aux[14]);/*p12o12n12m12p8o8n8m8p4o4n4m4p0o0n0m0*/
        aux[14] = _mm_unpackhi_epi16(aux[12],aux[14]);/*p14o14n14m14p10o10n10m10p6o6n6m6p2o2n2m2*/
        aux[12] = a;

        a = _mm_unpacklo_epi16(aux[13],aux[15]);/*p13o13n13m13p9o9n9m9p5o5n5m5p1o1n1m1*/
        aux[15] = _mm_unpackhi_epi16(aux[13],aux[15]);/*p15o15n15m15p11o11n11m11p7o7n7m7p3o3n3m3*/
        aux[13] = a;

/*------------------------------------------------------------------------------------------------------------*/
        
        a = _mm_unpacklo_epi32(aux[0],aux[4]);/*h08g08f08e08d08c08b08a08h00g00f00e00d00c00b00a00*/
        aux[4] = _mm_unpackhi_epi32(aux[0],aux[4]);/*h12g12f12e12d12c12b12a12h04g04f04e04d04c04b04a04*/
        aux[0] = a;

        a = _mm_unpacklo_epi32(aux[1],aux[5]);/*h09g09f09e09d09c09b09a09h01g01f01e01d01c01b01a01*/
        aux[5] = _mm_unpackhi_epi32(aux[1],aux[5]);/*h13g13f13e13d13c13b13a13h05g05f05e05d05c05b05a05*/
        aux[1] = a;

        a = _mm_unpacklo_epi32(aux[2],aux[6]);/*h10g10f10e10d10c10b10a10h02g02f02e02d02c02b02a02*/
        aux[6] = _mm_unpackhi_epi32(aux[2],aux[6]);/*h14g14f14e14d14c14b14a14h06g06f06e06d06c06b06a06*/
        aux[2] = a;

        a = _mm_unpacklo_epi32(aux[3],aux[7]);/*h11g11f11e11d11c11b11a11h03g03f03e03d03c03b03a03*/
        aux[7] = _mm_unpackhi_epi32(aux[3],aux[7]);/*h15g15f15e15d15c15b15a11h07g07f07e07d07c07b07a07*/
        aux[3] = a;

        a = _mm_unpacklo_epi32(aux[8],aux[12]);/*p08o08n08m08l08k08j08i08p00o00n00m00l00k00j00i00*/
        aux[12] = _mm_unpackhi_epi32(aux[8],aux[12]);/*p12o12n12m12l12k12j12i12p04o04n04m04l04k04j04i04*/
        aux[8] = a;

        a = _mm_unpacklo_epi32(aux[9],aux[13]);/*p09o09n09m09l09k09j09i09p01o01n01m01l01k01j01i01*/
        aux[13] = _mm_unpackhi_epi32(aux[9],aux[13]);/*p13o13n13m13l13k13j13i13p05o05n05m05l05k05j05i05*/
        aux[9] = a;

        a = _mm_unpacklo_epi32(aux[10],aux[14]);/*p10o10n10m10l10k10j10i10p02o02n02m02l02k02j02i02*/
        aux[14] = _mm_unpackhi_epi32(aux[10],aux[14]);/*p14o14n14m14l14k14j14i14p06o06n06m06l06k06j06i06*/
        aux[10] = a;

        a = _mm_unpacklo_epi32(aux[11],aux[15]);/*p11o11n11m11l11k11j11i11p03o03n03m03l03k03j03i03*/
        aux[15] = _mm_unpackhi_epi32(aux[11],aux[15]);/*p15o15n15m15l15k15j15i11p07o07n07m07l07k07j07i07*/
        aux[11] = a;

/*------------------------------------------------------------------------------------------------------------*/
        a = _mm_unpacklo_epi64(aux[0],aux[8]);/*p00o00n00m00l00k00j00i00h00g00f00e00d00c00b00a00*/
        aux[8] = _mm_unpackhi_epi64(aux[0],aux[8]);/*p08o08n08m08l08k08j08i08h08g08f08e08d08c08b08a08*/
        aux[0] = a;

        a = _mm_unpacklo_epi64(aux[1],aux[9]);/*p01o01n01m01l01k01j01i01h01g01f01e01d01c01b01a01*/
        aux[9] = _mm_unpackhi_epi64(aux[1],aux[9]);/*p09o09n09m09l09k09j09i09h09g09f09e09d09c09b09a09*/
        aux[1] = a;

        a = _mm_unpacklo_epi64(aux[2],aux[10]);/*p02o02n02m02l02k02j02i02h02g02f02e02d02c02b02a02*/
        aux[10] = _mm_unpackhi_epi64(aux[2],aux[10]);/*p10o10n10m10l10k10j10i10h10g10f10e10d10c10b10a10*/
        aux[2] = a;
        
        a = _mm_unpacklo_epi64(aux[3],aux[11]);/*p03o03n03m03l03k03j03i03h03g03f03e03d03c03b03a03*/
        aux[11] = _mm_unpackhi_epi64(aux[3],aux[11]);/*p11o11n11m11l11k11j11i11h11g11f11e11d11c11b11a11*/
        aux[3] = a;

        a = _mm_unpacklo_epi64(aux[4],aux[12]);/*p04o04n04m04l04k04j04i04h04g04f04e04d04c04b04a04*/
        aux[12] = _mm_unpackhi_epi64(aux[4],aux[12]);/*p12o12n12m12l12k12j12i12h12g12f12e12d12c12b12a12*/
        aux[4] = a;

        a = _mm_unpacklo_epi64(aux[5],aux[13]);/*p05o05n05m05l05k05j05i05h05g05f05e05d05c05b05a05*/
        aux[13] = _mm_unpackhi_epi64(aux[5],aux[13]);/*p13o13n13m13l13k13j13i13h13g13f13e13d13c13b13a13*/
        aux[5] = a;

        a = _mm_unpacklo_epi64(aux[6],aux[14]);/*p06o06n06m06l06k06j06i06h06g06f06e06d06c06b06a06*/
        aux[14] = _mm_unpackhi_epi64(aux[6],aux[14]);/*p14o14n14m14l14k14j14i14h14g14f14e14d14c14b14a14*/
        aux[6] = a;

        a = _mm_unpacklo_epi64(aux[7],aux[15]);/*p07o07n07m07l07k07j07i07h07g07f07e07d07c07b07a07*/
        aux[15] = _mm_unpackhi_epi64(aux[7],aux[15]);/*p15o15n15m15l15k15j15i11h15g15f15e15d15c15b15a11*/
        aux[7] = a;
/*------------------------------------------------------------------------------------*/

        output[c] = _mm_setzero_si128();
        for(j=0;j<16;j++)
            output[c] = _mm_xor_si128(output[c],aux[j]);

        if(i==143){
            break;
        }else if(i > 128){
            c++;
            i = c;
        }
    }

}

static void substitution(__m128i *r) {
    __m128i r1,r2,a,b,c;


    c = r[0];
    b = r[1];
    a = r[2];
    r1 = _mm_xor_si128(a,_mm_and_si128(b,c));
    r2 = _mm_xor_si128(_mm_xor_si128(a,b), _mm_and_si128(a,c));
    r[0] = _mm_xor_si128(_mm_xor_si128(a,b), _mm_xor_si128(c, _mm_and_si128(a,b)));
    r[1] = r2;
    r[2] = r1;

    c = r[3];
    b = r[4];
    a = r[5];
    r1 = _mm_xor_si128(a,_mm_and_si128(b,c));
    r2 = _mm_xor_si128(_mm_xor_si128(a,b), _mm_and_si128(a,c));
    r[3] = _mm_xor_si128(_mm_xor_si128(a,b), _mm_xor_si128(c, _mm_and_si128(a,b)));
    r[4] = r2;
    r[5] = r1;

    c = r[6];
    b = r[7];
    a = r[8];
    r1 = _mm_xor_si128(a,_mm_and_si128(b,c));
    r2 = _mm_xor_si128(_mm_xor_si128(a,b), _mm_and_si128(a,c));
    r[6] = _mm_xor_si128(_mm_xor_si128(a,b), _mm_xor_si128(c, _mm_and_si128(a,b)));
    r[7] = r2;
    r[8] = r1;    

}


void LowMCEnc(uint8_t in[][16], uint8_t out[][16], uint8_t key[][16])
{
    int i;
    ALIGN __m128i rKey[9];
    ALIGN __m128i rOut[9];
    ALIGN __m128i rRoundKey[9];


    if (in != out) {
        for(i=0;i<9;i++)
	        rOut[i] =_mm_load_si128((__m128i*)in[i+0]);
    }

    for(i=0;i<9;i++){
        rKey[i] =_mm_load_si128((__m128i*)key[i+0]);
    }

    matrix_mulBitslice(rRoundKey, rKey, KMatrixBitslice(0, 129));    
      for(i = 0;i<9;i++)
        rOut[i] = _mm_xor_si128(rOut[i],rRoundKey[i]);
        

    for (uint32_t r = 1; r <= ROUNDS; r++) {
        matrix_mulBitslice(rRoundKey, rKey, KMatrixBitslice(r, 129));
        substitution(rOut);   
        matrix_mulBitslice(rOut, rOut, LMatrixBitslice(r-1, 129)); 
        
        for(i = 0;i<9;i++){
            rOut[i] = _mm_xor_si128(rOut[i],_mm_load_si128((__m128i*)ConstantBitslice(r-1,129) + (i)));
        }

         for(i = 0;i<9;i++)
            rOut[i] = _mm_xor_si128(rOut[i],rRoundKey[i]);              
    }
    
    for(i=0;i<9;i++){
	    _mm_store_si128((__m128i*)out[i],rOut[i]);
    }
}


void LowMCEnc2(uint8_t in_t[][17], uint8_t out_t[][17], uint8_t key_t[][17])
{
    int i;
    ALIGN __m128i rKey[9];
    ALIGN __m128i rOut[9];
    ALIGN __m128i rRoundKey[9];

    ALIGN uint8_t in[9][16] = {0};
    ALIGN uint8_t aux[8][17] = {0};
    ALIGN uint8_t key[9][16] = {0};

    ALIGN uint8_t out[9][16] = {0};

    stateTrans(&in_t[0],aux);
    stateTrans2(&aux[0],in);

    stateTrans(&key_t[0],aux);
    stateTrans2(&aux[0],key);

    if (in != out) {
        for(i=0;i<9;i++)
	        rOut[i] =_mm_load_si128((__m128i*)in[i+0]);
    }

    for(i=0;i<9;i++){
        rKey[i] =_mm_load_si128((__m128i*)key[i+0]);
    }

    matrix_mulBitslice(rRoundKey, rKey, KMatrixBitslice(0, 129));    

      for(i = 0;i<9;i++)
        rOut[i] = _mm_xor_si128(rOut[i],rRoundKey[i]);
        

    for (uint32_t r = 1; r <= ROUNDS; r++) {
        matrix_mulBitslice(rRoundKey, rKey, KMatrixBitslice(r, 129));
        substitution(rOut);        
        matrix_mulBitslice(rOut, rOut, LMatrixBitslice(r-1, 129)); 
        
        for(i = 0;i<9;i++){
            rOut[i] = _mm_xor_si128(rOut[i],_mm_load_si128((__m128i*)ConstantBitslice(r-1,129) + (i)));
        }

         for(i = 0;i<9;i++)
            rOut[i] = _mm_xor_si128(rOut[i],rRoundKey[i]);              
    }
    
    for(i=0;i<9;i++){
	    _mm_store_si128((__m128i*)out[i],rOut[i]);
    }

    stateTrans2Back(&out[0],aux);
    stateTransBack(&aux[0] , out_t);

}





int main(){
      ALIGN uint8_t key[8][17] = {\
      {0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
      {0xab, 0x22, 0x42, 0x51, 0x49, 0xaa, 0x61, 0x2d, 0x7f, 0xff, 0x13, 0x72, 0x20, 0x27, 0x5b, 0x16, 0x80},
      {0xe7, 0x3a, 0xf2, 0x9c, 0xfc, 0x7a, 0xe5, 0x3e, 0x52, 0x20, 0xd3, 0x1e, 0x2e, 0x59, 0x17, 0xda, 0x80},
      {0x30, 0xf3, 0x34, 0x88, 0x53, 0x2d, 0x7e, 0xb8, 0xa5, 0xf8, 0xfb, 0x4f, 0x2e, 0x63, 0xba, 0x56, 0x00},
      {0x80, 0xAB, 0x5d, 0x8b, 0x08, 0x90, 0xA0, 0x50, 0x20, 0x30, 0x40, 0x80, 0x10, 0xf0, 0xff, 0x55, 0xcc},
      {0xEF, 0x10, 0xfd, 0x04, 0x83, 0x04, 0xAf, 0x5d, 0x21, 0x32, 0x43, 0x84, 0x15, 0xf6, 0xf7, 0x58, 0xc9},
      {0xb0, 0x71, 0xc6, 0xd4, 0xa3, 0x77, 0xe5, 0x51, 0x25, 0x4c, 0x5d, 0xc4, 0x1,  0xa3, 0xd0, 0x8a, 0xcb},
      {0x44, 0x48, 0xc7, 0x0a, 0xc3, 0x86, 0x30, 0x21, 0xbe, 0x23, 0x2c, 0x63, 0x38, 0x16, 0x87, 0xcd, 0x5d}
  };

    ALIGN uint8_t plaintext[8][17] = {
      {0xab, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
      {0x4b, 0x99, 0x23, 0x53, 0xa6, 0x6,  0x65, 0xbf, 0x99, 0x2d, 0x3,  0x54, 0x82, 0xc1, 0xd2, 0x79, 0x00},
      {0x30, 0x4b, 0xa7, 0xa8, 0xde, 0x2b, 0x5c, 0xf8, 0x87, 0xf9, 0xa4, 0x8a, 0xb7, 0x56, 0x1b, 0xf6, 0x80},
      {0xc2, 0x6a, 0x5d, 0xf9, 0x6,  0x15, 0x8d, 0xcb, 0x6a, 0xc7, 0x89, 0x1d, 0xa9, 0xf4, 0x9f, 0x78, 0x00},
      {0xab, 0xff, 0xdd, 0xee, 0xff, 0x11, 0x22, 0x12, 0x13, 0x51, 0x67, 0xfd, 0xbc, 0x12, 0x34, 0x98, 0x8d},
      {0xBD, 0xf1, 0xd2, 0xe3, 0xf4, 0x15, 0x26, 0x17, 0x18, 0x59, 0x6a, 0xfb, 0xbd, 0x1c, 0x3e, 0x9f, 0x8f},
      {0x99, 0x60, 0x9f, 0x41, 0x8a, 0x6B, 0x2E, 0x1C, 0xE8, 0xF9, 0x43, 0x0F, 0xF5, 0xCF, 0x73, 0x17, 0x9D},
      {0xef, 0xb5, 0x0b, 0xa2, 0x8d, 0x7b, 0x26, 0x8e, 0x19, 0x72, 0x7b, 0xae, 0xbc, 0x67, 0x9a, 0x66, 0x7c}
  };
    ALIGN uint8_t ciphertext_expected[8][17] = {\
      {0x2f, 0xd7, 0xd5, 0x42, 0x5e, 0xe3, 0x5e, 0x66, 0x7c, 0x97, 0x2f, 0x12, 0xfb, 0x15, 0x3e, 0x9d, 0x80},
      {0x2a, 0x40, 0x62, 0xd8, 0x35, 0xc5, 0x93, 0xea, 0x19, 0xf8, 0x22, 0xad, 0x24, 0x24, 0x77, 0xd2, 0x80},
      {0x5c, 0xd2, 0xc3, 0x55, 0x32, 0x8e, 0xfd, 0xe9, 0xf3, 0x78, 0xc1, 0x61, 0x23, 0xd3, 0x3f, 0xb3, 0x00},
      {0x0b, 0x43, 0xb6, 0x5f, 0x7c, 0x53, 0x50, 0x06, 0xcf, 0x27, 0xe8, 0x6f, 0x55, 0x1b, 0xd0, 0x15, 0x80},
      {0x7D, 0xE8, 0xA8, 0xB2, 0xE5, 0x0A, 0xDB, 0x76, 0x9D, 0xEA, 0x99, 0xBD, 0x34, 0xA0, 0x5D, 0x4F, 0x80},
      {0xC7, 0x8D, 0xC4, 0x9E, 0xD0, 0xCC, 0x1E, 0xAA, 0xD7, 0x90, 0xCE, 0x44, 0x6F, 0xF3, 0x95, 0x54, 0x80},
      {0xCA, 0x4D, 0x63, 0x46, 0x61, 0x3F, 0x0A, 0x5B, 0x44, 0xC0, 0xA9, 0xED, 0xB1, 0xA8, 0x00, 0xD0, 0x80},
      {0x5E, 0x05, 0x0A, 0x9F, 0x73, 0x09, 0x02, 0x03, 0x8B, 0x9D, 0x1E, 0x49, 0xB1, 0x5A, 0x91, 0x0E, 0x80}
    };


    ALIGN uint8_t plaintextAux[9][16] = {0};
    ALIGN uint8_t aux[8][17] = {0};
    ALIGN uint8_t keyAux[9][16] = {0};

    ALIGN uint8_t out[8][17] = {0};
    ALIGN uint8_t outAux[9][16] = {0};
    stateTrans(&plaintext[0],aux);/*8x17 -> 8 x 17*/
    stateTrans2(&aux[0],plaintextAux); /*8x17 -> 9x16*/

    stateTrans(&key[0],aux);
    stateTrans2(&aux[0],keyAux);

    LowMCEnc(plaintextAux, outAux, keyAux);
    stateTrans2Back(&outAux[0], aux);
    stateTransBack(&aux[0], out);
    
    int res = 0;
    for(int j=0;j<8;j++){
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
    // BENCH_FUNCTION(LowMCEnc, plaintextAux, outAux, keyAux);

    // printf("Completa!\n\n");

    LowMCEnc2(plaintext, out, key);

    res = 0;
    for(int j=0;j<8;j++){
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
    BENCH_FUNCTION(LowMCEnc2, plaintext, out, key);
   

    printBytes(out[0],0);
    return 0;        
}



