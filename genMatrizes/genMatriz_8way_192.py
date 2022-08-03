blocksize = 192
keysize = 192
rounds = 4

mat = [['0x00' for x in range(192)] for y in range(768)] 
mat2 = [['0x00' for x in range(192)] for y in range(768)] 

def orgLinlayers(m):
    x = 0
    for a in m:
        for b in a:
            y = 0
            for c in b:
                if (c == 1):
                    mat[x][y] = '0xff'
                else:
                    mat[x][y] = '0x00'
                y = y + 1
            x = x+1
    
    for i in range (768):
        j = 0
        k = 0
        c = 0
        while(1):
            # print (k, j)
            mat2[i][k] = mat[i][j]
            j = (j + 12)
            k = k + 1
            if(j > 191):
                c = c + 1
                j = c
            if(c == 12):
                break
    # print (m)            
    print (mat2)

matRoundKey = [['0x00' for x in range(192)] for y in range(960)] 
mat2RoundKey = [['0x00' for x in range(192)] for y in range(960)] 
    

def orgRoundkey(m):
    x = 0
    for a in m:
        for b in a:
            y = 0
            for c in b:
                if (c == 1):
                    matRoundKey[x][y] = '0xff'
                else:
                    matRoundKey[x][y] = '0x00'
                y = y + 1
            x = x+1
    
    for i in range (192):
        j = 0
        k = 0
        c = 0
        while(1):
            print (k, j)
            mat2RoundKey[i][k] = matRoundKey[i][j]
            j = (j + 12)
            k = k + 1
            if(j > 191):
                c = c + 1
                j = c
            if(c == 12):
                break
    
    # print (m)        
    print (mat2RoundKey)


matRoundConstants = [['0x00' for x in range(192)] for y in range(4)] 
mat2RoundConstants = [['0x00' for x in range(192)] for y in range(4)] 


def orgRoundConstants(m):
    x = 0
    for a in m:
        y = 0
        for c in a:
            if (c == 1):
                matRoundConstants[x][y] = '0xff'
            else:
                matRoundConstants[x][y] = '0x00'
            y = y + 1
        x = x+1
    
    for i in range (4):
        j = 0
        k = 0
        c = 0
        while(1):
            # print (k, j)
            mat2RoundConstants[i][k] = matRoundConstants[i][j]
            j = (j + 12)
            k = k + 1
            if(j > 191):
                c = c + 1
                j = c
            if(c == 12):
                break
            
    print (mat2RoundConstants)


    # for x in range (129):
    #         for y in range(129):
    #             mat2[x][y] = mat[x].index(y)
    # print (mat2)            


def main():
    ''' Use the global parameters `blocksize`, `keysize` and `rounds`
        to create the set of matrices and constants for the corresponding
        LowMC instance. Save those in a file named
        `matrices_and_constants.dat`.
    '''
    gen = grain_ssg()
    linlayers = []
    for _ in range(rounds):
        linlayers.append(instantiate_matrix(blocksize, blocksize, gen))
    # orgLinlayers(linlayers)

    round_constants = []
    for _ in range(rounds):
        constant = [next(gen) for _ in range(blocksize)]
        round_constants.append(constant)
    # orgRoundConstants(round_constants)

    roundkey_matrices = []
    for _ in range(rounds + 1):
        mat = instantiate_matrix(blocksize, keysize, gen)
        roundkey_matrices.append(mat)
    orgRoundkey(roundkey_matrices)

    with open('matrices_and_constants.dat', 'w') as matfile:
        s = 'LowMC matrices and constants\n'\
            '============================\n'\
            'Block size: ' + str(blocksize) + '\n'\
            'Key size: ' + str(keysize) + '\n'\
            'Rounds: ' + str(rounds) + '\n\n'
        matfile.write(s)
        s = 'Linear layer matrices\n'\
            '---------------------'
        matfile.write(s)
        for r in range(rounds):
            s = '\nLinear layer ' + str(r + 1) + ':\n'
            for row in linlayers[r]:
                # org(row)
                s += str(row) + '\n'
            matfile.write(s)

        s = '\nRound constants\n'\
              '---------------------'
        matfile.write(s)
        for r in range(rounds):
            s = '\nRound constant ' + str(r + 1) + ':\n'
            s += str(round_constants[r]) + '\n'
            matfile.write(s)

        s = '\nRound key matrices\n'\
              '---------------------'
        matfile.write(s)
        for r in range(rounds + 1):
            s = '\nRound key matrix ' + str(r) + ':\n'
            for row in roundkey_matrices[r]:
                s += str(row) + '\n'
            matfile.write(s)

def instantiate_matrix(n, m, gen):
    ''' Instantiate a matrix of maximal rank using bits from the
        generatator `gen`.
    '''
    while True:
        mat = []
        for _ in range(n):
            row = []
            for _ in range(m):
                row.append(next(gen))
            mat.append(row)
        if rank(mat) >= min(n, m):
            return mat

def rank(matrix):
    ''' Determine the rank of a binary matrix. '''
    # Copy matrix
    mat = [[x for x in row] for row in matrix]
    
    n = len(matrix)
    m = len(matrix[0])
    for c in range(m):
        if c > n - 1:
            return n
        r = c
        while mat[r][c] != 1:
            r += 1
            if r >= n:
                return c
        mat[c], mat[r] = mat[r], mat[c]
        for r in range(c + 1, n):
            if mat[r][c] == 1:
                for j in range(m):
                    mat[r][j] ^= mat[c][j]
    return m


def grain_ssg():
    ''' A generator for using the Grain LSFR in a self-shrinking generator. '''
    state = [1 for _ in range(80)]
    index = 0
    # Discard first 160 bits
    for _ in range(160):
        state[index] ^= state[(index + 13) % 80] ^ state[(index + 23) % 80]\
                        ^ state[(index + 38) % 80] ^ state[(index + 51) % 80]\
                        ^ state[(index + 62) % 80]
        index += 1
        index %= 80
    choice = False
    while True:
        state[index] ^= state[(index + 13) % 80] ^ state[(index + 23) % 80]\
                        ^ state[(index + 38) % 80] ^ state[(index + 51) % 80]\
                        ^ state[(index + 62) % 80]
        choice = state[index]
        index += 1
        index %= 80
        state[index] ^= state[(index + 13) % 80] ^ state[(index + 23) % 80]\
                        ^ state[(index + 38) % 80] ^ state[(index + 51) % 80]\
                        ^ state[(index + 62) % 80]
        if choice == 1:
            yield state[index]
        index += 1
        index %= 80


if __name__ == '__main__':
    main()
