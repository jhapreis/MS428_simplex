import numpy as np
import pandas as pd
from simplex_primal import MetodoSimplexPrimal


class MetodoSimplexDuasFases(MetodoSimplexPrimal):      

    def __find_identity_array_in_original_matrix(self) -> dict:

        eye = np.eye(self.A.shape[0])

        identity_indexes = []
        array_indexes = []

        seen = set()

        for i in range(self.A.shape[1]):
            
            for j in range(self.A.shape[0]):
            
                if np.all( self.A[:, i] == eye[:, j] ):
                    
                    if j not in seen:
                        
                        seen.add(j)
                        
                        array_indexes.append(i)
                        identity_indexes.append(j)
                                    
                        break

        order_dict = dict(
            map(
                lambda x, y: (x,y),
                identity_indexes,
                array_indexes
            )
        )
        
        return order_dict


    def __complete_coefficients_as_index_matrix(self, order_dict: dict) -> tuple:
        
        eye = np.eye(self.A.shape[0])
        
        A_complete = self.A

        if len(order_dict) < self.A.shape[0]:

            for i in range(self.A.shape[0]):
                
                if order_dict.get(i) is None:
                    
                    A_complete = np.hstack(  ( A_complete, eye[:, [i]] )  )
        
        num_completed_columns = self.A.shape[0] - len(order_dict)
        
        X_complete = np.hstack((  
            self.X,  
            np.array([  f"y_{i+1}" for i in range(num_completed_columns)  ])  
        ))
                        
        C_complete = np.vstack((
            self.C,
            np.zeros((num_completed_columns, 1))
        ))
                
        return A_complete, X_complete, C_complete
    
    
    def __split_particoes_basica_nao_basica(self, order_dict: dict):
        
        total  = set(range(self.A.shape[1]))
        
        eye_in_A = [order_dict.get(i) for i in range(len(order_dict))]
        
        eye_completed = [i+self.A.shape[1] for i in range(self.A.shape[0] - len(order_dict))]
        
        eye_in_A.extend(eye_completed)
                
        part_b = set(eye_in_A)

        part_n = total - part_b 

        return list(part_b), list(part_n)


    def solve(self) -> pd.DataFrame:
                
        order_dict = self.__find_identity_array_in_original_matrix()
        
        A_complete, X_complete, C_complete = self.__complete_coefficients_as_index_matrix(order_dict=order_dict)
                
        part = self.__split_particoes_basica_nao_basica(order_dict=order_dict)
        
        self.A = A_complete
        self.C = C_complete
        self.X = X_complete
        
        df_sol = self._simplex(particao=part)

        return df_sol
