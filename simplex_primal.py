from abc import abstractmethod
import numpy as np
import pandas as pd
from random import sample


class AbstractSimplex:
    
    @abstractmethod
    def solve(self) -> np.ndarray:
        pass


class MetodoSimplexPrimal(AbstractSimplex):
    
    def __init__(self, A, b, C) -> None:

        self.A = A
        self.b = b
        self.C = C
        self.X = np.array([
            f"x_{i+1}" for i in range(self.A.shape[1])
        ])
        
        
    def _solucao_basica(self, B: np.ndarray) -> np.ndarray:

        solucao = np.linalg.solve(B,self.b)

        return solucao


    def _verifica_solucao_basica_factivel(self, solucao_basica: np.ndarray) -> bool:

        return np.all(solucao_basica >= 0)
    

    def _custos_relativos(self, B: np.ndarray, N: np.ndarray, C_B: np.ndarray, C_N: np.ndarray) -> np.ndarray:
        
        lambda_value = np.linalg.solve( B.T, C_B )
        
        print(lambda_value.shape)
        print(N.shape)
        print(C_N.shape)
        
        custos = C_N.T - (lambda_value.T @ N)
        
        print(custos.shape)
        
        return custos
    

    def _verifica_condicao_otima(self, custos: np.ndarray) -> bool:

        return np.all(custos >= 0)
    

    def _formata_solucao_basica_como_dataframe(self, solucao_basica_factivel: np.ndarray, x_b: np.ndarray, x_n: np.ndarray) -> pd.DataFrame:

        df = pd.DataFrame( 
            data= np.append(solucao_basica_factivel,np.array([0,0])),
            index=np.append(x_b,x_n),
            columns=['value']
        )

        return df
    

    def _index_out_particao_basica(self, B: np.ndarray, N: np.ndarray, index_in: int, solucao_basica: np.ndarray) -> int:

        y = np.linalg.solve(  B, N[:,[index_in]]  )

        if np.all(y <= 0):

            return -1

        epsilons = []

        for i in range(y.shape[0]):

            if y[i] > 0:
                epsilons.append(solucao_basica[i][0]/y[i][0])

            else:
                epsilons.append(np.inf)

        minimal = np.array(epsilons).argmin()

        return minimal
    

    def _troca_colunas_particoes(self, B: np.ndarray, N: np.ndarray, C_B: np.ndarray, C_N: np.ndarray, x_b: np.ndarray, x_n: np.ndarray, index_out: int, index_in: int) -> None:

        coluna_out = np.copy( B[:,[index_out]] )

        x_out = np.copy(x_b[index_out])

        c_out = np.copy(C_B[index_out])

        x_b[index_out] = x_n[index_in]

        x_n[index_in] = x_out

        B[:,[index_out]] = N[:,[index_in]]

        N[:,[index_in]]  = coluna_out

        C_B[index_out] = C_N[index_in]

        C_N[index_in]  = c_out
        

    def __seleciona_particao_random(self, num_linhas: int, num_colunas: int):

        total = set(range(num_colunas))

        part_b = set(sample(total, num_linhas))

        part_n = total - part_b 

        return list(part_b), list(part_n)
    
    
    def _simplex(self, particao: list):
        
        C_B, C_N, B, N, x_b, x_n = self._particiona_basico_nao_basico(particao)
        
        solucao_basica_factivel = self._solucao_basica(B)
        
        otimo   = False

        counter = 0

        while not otimo:

            print(f"\nB\n{B}\n")
            print(f"N\n{N}\n")
            print(f"C_B\n{C_B}\n")
            print(f"C_N\n{C_N}\n")

            counter += 1

            custos = self._custos_relativos(B, N, C_B, C_N)

            print(f"custos: {custos}\n")

            otimo = self._verifica_condicao_otima(custos=custos)

            if otimo == True:

                print("Solucao otima atingida!")

                df = self._formata_solucao_basica_como_dataframe(solucao_basica_factivel, x_b, x_n)
                print(df)
                
                break

            index_in  = custos.argmin()

            index_out = self._index_out_particao_basica(B, N, index_in, solucao_basica_factivel)

            if index_out == -1:

                print("Problema nao tem solucao finita otima")

                break

            print(f"Trocando colunas \"{x_b[index_out]}\" sai da base; \"{x_n[index_in]}\" entra.\n")
            self._troca_colunas_particoes(B, N, C_B, C_N, x_b, x_n, index_out, index_in)

            print(f"B\n{B}\n")
            print(f"N\n{N}\n")
            print(f"C_B\n{C_B}\n")
            print(f"C_N\n{C_N}\n")

            solucao_basica_factivel = self._solucao_basica(B)

            df = self._formata_solucao_basica_como_dataframe(solucao_basica_factivel, x_b, x_n)
            print(f"\n----- Particao basica: -----\n{df}")

        print(f"\n\nApos {counter} tentativa(s).")

        return df


    def _particiona_basico_nao_basico(self, particao: list):
        
        C_B = self.C[particao[0], :]
        C_N = self.C[particao[1], :]
        B   = self.A[:, particao[0]]
        N   = self.A[:, particao[1]]
        x_b = self.X[particao[0]]
        x_n = self.X[particao[1]]
        
        return C_B, C_N, B, N, x_b, x_n


    def solve(self) -> pd.DataFrame:

        counter = 0

        particao_eh_factivel = False

        while not particao_eh_factivel:

            part = self.__seleciona_particao_random(*self.A.shape)

            C_B, C_N, B, N, x_b, x_n = self._particiona_basico_nao_basico(part)

            solucao_basica_factivel = self._solucao_basica(B)

            particao_eh_factivel    = self._verifica_solucao_basica_factivel(solucao_basica_factivel)

            counter += 1

        print(f"Particao basica factivel encontrada apos {counter} tentativa(s).")

        df = self._formata_solucao_basica_como_dataframe(solucao_basica_factivel, x_b, x_n)
        print(f"\n----- Particao basica: -----\n{df}")

        df_sol = self._simplex(part)
        
        return df_sol
