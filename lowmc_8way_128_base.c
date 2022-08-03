/*! @file lowmc_8way_128_base.c
 *  @brief Base code of the 8-way implementation of the LowMC-129-129-4   
 *  The code is provided under the GPL license, see LICENSE for
 *  more details.
  */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "x86intrin.h"
#include "immintrin.h"
#include "emmintrin.h"
#include "constants.h"


#include "bench.h"


#define BENCH 1000


#define PARALLEL    8
#define ROUNDS       4
#define loadLinha(i,k)          aux[k] =  _mm_setzero_si128();\
                                for(j=0;j<9;j++){\
                                    temp = _mm_load_si128((__m128i*)matrix+(i*9+j));\
                                    temp = _mm_and_si128(temp,r[j]);\
                                    aux[k] = _mm_xor_si128(aux[k],temp);\
                                }\
                               

uint64_t transpose8(uint64_t x) {

    x = (x & 0xAA55AA55AA55AA55LL) | ((x & 0x00AA00AA00AA00AALL) << 7)  | ((x >> 7) & 0x00AA00AA00AA00AALL);
    x = (x & 0xCCCC3333CCCC3333LL) | ((x & 0x0000CCCC0000CCCCLL) << 14) | ((x >> 14) & 0x0000CCCC0000CCCCLL);
    x = (x & 0xF0F0F0F00F0F0F0FLL) | ((x & 0x00000000F0F0F0F0LL) << 28) | ((x >> 28) & 0x00000000F0F0F0F0LL);
    
    return x;

} 

void stateTransBack(uint8_t in[][17], uint8_t out[][17]){
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
    out[0][16] = comp;//destacar no texto.
}

void stateTrans2(uint8_t in[][17], uint8_t out[][16]){
    int i,t,j,k;
    i = 0; j = 0; k = 0; t = 0;
    while(1){
        out[t][k] = in[i][j];
        i++; t++;
        if(i == 8){
            i = 0; j++;
        }
        if(t == 9){
            t = 0; k++;
        }
        if(k == 14)
            break;
    }
    out[0][14] = in [6][15];
    out[1][14] = in [7][15];
    out[2][14] = in [0][16];
    out[3][14] = 0;
    out[4][14] = 0;
    out[5][14] = 0;
    out[6][14] = 0;
    out[7][14] = 0;
    out[8][14] = 0;
}


void stateTrans2Back(uint8_t in[][16], uint8_t out[][17]){
    int i,t,j,k;
    i = 0; j = 0; k = 0; t = 0;
    while(1){
        out[i][j] = in[t][k];
        i++; t++;
        if(i == 8){
            i = 0; j++;
        }
        if(t == 9){
            t = 0; k++;
        }
        if(k == 14)
            break;
    }
    out[6][15] = in[0][14];
    out[7][15] = in[1][14];
    out[0][16] = in[2][14];
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
//     printf("%8.8x%8.8x%8.8x%8.8x\n", c[3],c[2],c[1],c[0]);
// }


void matrix_mulBitslice(
    __m128i *output,
    __m128i *rIn,
    const uint8_t* matrix)
{
    __m128i aux[16], aux2[16], r[9], temp;

    int i,j,c,k;

    for(i=0;i<9;i++){
	    r[i] = rIn[i];
    }
    
    i = 0;
    c = 0;
    
    while(1){
        k = 0;

        for(k=0;k<14;k++){
            loadLinha(i,k)
            i+=9;
        }

        if(c < 3){
            loadLinha(i,14)
            i+=9;
            aux[15] = _mm_setzero_si128();
            
        }else{
            i+=9;
            aux2[14] = _mm_setzero_si128();
            aux2[15] = _mm_setzero_si128();
        }

      

        for(k=0;k<16;k+=2){
            aux2[k] = _mm_unpacklo_epi8(aux[k],aux[k+1]);  
            aux2[k+1] = _mm_unpackhi_epi8(aux[k],aux[k+1]);
        }

        for(k=0,j=1;k<16;k+=4,j+=4){
            aux[k] = _mm_unpacklo_epi16(aux2[k],aux2[k+2]);     
            aux[k+2] = _mm_unpackhi_epi16(aux2[k],aux2[k+2]);   
            aux[j]= _mm_unpacklo_epi16(aux2[j],aux2[j+2]);      
            aux[j+2] = _mm_unpackhi_epi16(aux2[j],aux2[j+2]);   
        }

        for(k=0,j=8;k<4;k++,j++){
            aux2[k] = _mm_unpacklo_epi32(aux[k],aux[k+4]);      
            aux2[k+4] = _mm_unpackhi_epi32(aux[k],aux[k+4]);    
            aux2[j] = _mm_unpacklo_epi32(aux[j],aux[j+4]);      
            aux2[j+4] = _mm_unpackhi_epi32(aux[j],aux[j+4]);    
        }

        for(k=0;k<8;k++){
            aux[k] = _mm_unpacklo_epi64(aux2[k],aux2[k+8]);         
            aux[k+8] = _mm_unpackhi_epi64(aux2[k],aux2[k+8]);       
        }
        
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

    for(int i=0;i<3;i++){
        c = r[i*3 + 0];
        b = r[i*3 + 1];
        a = r[i*3 + 2];
        r1 = _mm_xor_si128(a,_mm_and_si128(b,c));
        r2 = _mm_xor_si128(_mm_xor_si128(a,b), _mm_and_si128(a,c));
        r[i*3 + 0] = _mm_xor_si128(_mm_xor_si128(a,b), _mm_xor_si128(c, _mm_and_si128(a,b)));
        r[i*3 + 1] = r2;
        r[i*3 + 2] = r1;
    }
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
    ALIGN uint8_t keyAux[9][16] = {0};

    ALIGN uint8_t out[9][17] = {0};
    ALIGN uint8_t outAux[9][16] = {0};

    stateTrans(&plaintext[0],plaintext);
    stateTrans2(&plaintext[0],plaintextAux);

    stateTrans(&key[0],key);
    stateTrans2(&key[0],keyAux);

    LowMCEnc(plaintextAux, outAux, keyAux);
    
    stateTrans2Back(&outAux[0],out);
    stateTransBack(&out[0] , out);



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

    stateTrans(&plaintext[0],plaintext);
    stateTrans2(&plaintext[0],plaintextAux);

    stateTrans(&key[0],key);
    stateTrans2(&key[0],keyAux);

    BENCH_FUNCTION(LowMCEnc, plaintextAux, outAux, keyAux);

    printBytes(out[0],0);
    return 0;        
}



