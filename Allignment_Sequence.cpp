
#include<iostream>

using namespace std;

int mele(int a,int b,int c )
{
    if(a>b)
    {
        if(a>c) return a;
        else return c;
    }
    else
    {
        if(b>c) return b;
        else return c;
    }
}


int main()
{
    string s1;
    cout<<"---Enter s1---\n";
    cin>>s1;
    string s2;
    cout<<"---Enter s1---\n";
    cin>>s2;
    int ma_sc = 2;
    int mis_sc = -1;
    
    int len1 = s1.length();
    int len2 = s2.length();
    
    int mat[len2+1][len1+1];
    
    for (int i = 0; i < len2+1; i++) 
    {
                mat[0][i] = 0;
        }
        
    for (int j = 0; j < len1+1; j++)
    {
        mat[j][0] = 0;
    }
    
    int max_score = 0;
    int max_i,max_j;
    
    for(int i=1;i<len2+1;i++)
    {            
        for(int j=1;j<len1+1;j++)
        {
            if(s1[j-1] == s2[i-1])
            {
                mat[i][j] = mat[i-1][j-1] +ma_sc;
                
            }
            else
            {
                mat[i][j] = mele(mat[i-1][j]+mis_sc,mat[i][j-1]+mis_sc,mat[i-1][j-1]+mis_sc);
            }
            
            if(mat[i][j] > max_score)
            {
                max_score = mat[i][j];
                max_i = i;
                max_j = j;
            }
        }
        
    }
    
    int i = max_i;
    int j = max_j;
    
    cout<<"ALLIGNMENT SEQUENCE \n";
    while((i!=0 && j!=0))
    {
        if(s1[j-1] == s2[i-1])
        {
            cout << s1[j-1] << '\t' << s2[i-1] << endl;
            i -= 1;
            j -= 1;
        }
        
        else
        {
            if(mat[i-1][j] > mat[i][j-1] && mat[i-1][j] > mat[i-1][j-1])
            {
                cout << '_' << '\t' << s2[i-1] << endl;
                i--;
            }
        
            else if(mat[i][j-1] > mat[i-1][j] && mat[i][j-1] > mat[i-1][j-1])
            {
                cout << s1[j-1] << '\t' << '_' << endl;
                j--;
            }
        }
    }
     cout<<"MATRIX \n";
     for (int i = 0; i < len2+1; i++) 
     {
        for (int j = 0; j < len1+1; j++) 
        {
            cout << mat[i][j] << '\t';
        }
        cout << endl;
    }
  
}

