{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# importing libraries\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt, seaborn as sns\n",
    "%matplotlib inline\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Dataset with rows 45211 and columns 19\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>job</th>\n",
       "      <th>salary</th>\n",
       "      <th>marital</th>\n",
       "      <th>education</th>\n",
       "      <th>targeted</th>\n",
       "      <th>default</th>\n",
       "      <th>balance</th>\n",
       "      <th>housing</th>\n",
       "      <th>loan</th>\n",
       "      <th>contact</th>\n",
       "      <th>day</th>\n",
       "      <th>month</th>\n",
       "      <th>duration</th>\n",
       "      <th>campaign</th>\n",
       "      <th>pdays</th>\n",
       "      <th>previous</th>\n",
       "      <th>poutcome</th>\n",
       "      <th>response</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>58</td>\n",
       "      <td>management</td>\n",
       "      <td>100000</td>\n",
       "      <td>married</td>\n",
       "      <td>tertiary</td>\n",
       "      <td>yes</td>\n",
       "      <td>no</td>\n",
       "      <td>2143</td>\n",
       "      <td>yes</td>\n",
       "      <td>no</td>\n",
       "      <td>unknown</td>\n",
       "      <td>5</td>\n",
       "      <td>may</td>\n",
       "      <td>261</td>\n",
       "      <td>1</td>\n",
       "      <td>-1</td>\n",
       "      <td>0</td>\n",
       "      <td>unknown</td>\n",
       "      <td>no</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>44</td>\n",
       "      <td>technician</td>\n",
       "      <td>60000</td>\n",
       "      <td>single</td>\n",
       "      <td>secondary</td>\n",
       "      <td>yes</td>\n",
       "      <td>no</td>\n",
       "      <td>29</td>\n",
       "      <td>yes</td>\n",
       "      <td>no</td>\n",
       "      <td>unknown</td>\n",
       "      <td>5</td>\n",
       "      <td>may</td>\n",
       "      <td>151</td>\n",
       "      <td>1</td>\n",
       "      <td>-1</td>\n",
       "      <td>0</td>\n",
       "      <td>unknown</td>\n",
       "      <td>no</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>33</td>\n",
       "      <td>entrepreneur</td>\n",
       "      <td>120000</td>\n",
       "      <td>married</td>\n",
       "      <td>secondary</td>\n",
       "      <td>yes</td>\n",
       "      <td>no</td>\n",
       "      <td>2</td>\n",
       "      <td>yes</td>\n",
       "      <td>yes</td>\n",
       "      <td>unknown</td>\n",
       "      <td>5</td>\n",
       "      <td>may</td>\n",
       "      <td>76</td>\n",
       "      <td>1</td>\n",
       "      <td>-1</td>\n",
       "      <td>0</td>\n",
       "      <td>unknown</td>\n",
       "      <td>no</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>47</td>\n",
       "      <td>blue-collar</td>\n",
       "      <td>20000</td>\n",
       "      <td>married</td>\n",
       "      <td>unknown</td>\n",
       "      <td>no</td>\n",
       "      <td>no</td>\n",
       "      <td>1506</td>\n",
       "      <td>yes</td>\n",
       "      <td>no</td>\n",
       "      <td>unknown</td>\n",
       "      <td>5</td>\n",
       "      <td>may</td>\n",
       "      <td>92</td>\n",
       "      <td>1</td>\n",
       "      <td>-1</td>\n",
       "      <td>0</td>\n",
       "      <td>unknown</td>\n",
       "      <td>no</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>33</td>\n",
       "      <td>unknown</td>\n",
       "      <td>0</td>\n",
       "      <td>single</td>\n",
       "      <td>unknown</td>\n",
       "      <td>no</td>\n",
       "      <td>no</td>\n",
       "      <td>1</td>\n",
       "      <td>no</td>\n",
       "      <td>no</td>\n",
       "      <td>unknown</td>\n",
       "      <td>5</td>\n",
       "      <td>may</td>\n",
       "      <td>198</td>\n",
       "      <td>1</td>\n",
       "      <td>-1</td>\n",
       "      <td>0</td>\n",
       "      <td>unknown</td>\n",
       "      <td>no</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   age           job  salary  marital  education targeted default  balance  \\\n",
       "0   58    management  100000  married   tertiary      yes      no     2143   \n",
       "1   44    technician   60000   single  secondary      yes      no       29   \n",
       "2   33  entrepreneur  120000  married  secondary      yes      no        2   \n",
       "3   47   blue-collar   20000  married    unknown       no      no     1506   \n",
       "4   33       unknown       0   single    unknown       no      no        1   \n",
       "\n",
       "  housing loan  contact  day month  duration  campaign  pdays  previous  \\\n",
       "0     yes   no  unknown    5   may       261         1     -1         0   \n",
       "1     yes   no  unknown    5   may       151         1     -1         0   \n",
       "2     yes  yes  unknown    5   may        76         1     -1         0   \n",
       "3     yes   no  unknown    5   may        92         1     -1         0   \n",
       "4      no   no  unknown    5   may       198         1     -1         0   \n",
       "\n",
       "  poutcome response  \n",
       "0  unknown       no  \n",
       "1  unknown       no  \n",
       "2  unknown       no  \n",
       "3  unknown       no  \n",
       "4  unknown       no  "
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Reading from CSV\n",
    "bm= pd.read_csv(\"bank-marketing.csv\")\n",
    "print(\"Dataset with rows {} and columns {}\".format(bm0.shape[0],bm0.shape[1]))\n",
    "bm.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 45211 entries, 0 to 45210\n",
      "Data columns (total 19 columns):\n",
      " #   Column     Non-Null Count  Dtype \n",
      "---  ------     --------------  ----- \n",
      " 0   age        45211 non-null  int64 \n",
      " 1   job        45211 non-null  object\n",
      " 2   salary     45211 non-null  int64 \n",
      " 3   marital    45211 non-null  object\n",
      " 4   education  45211 non-null  object\n",
      " 5   targeted   45211 non-null  object\n",
      " 6   default    45211 non-null  object\n",
      " 7   balance    45211 non-null  int64 \n",
      " 8   housing    45211 non-null  object\n",
      " 9   loan       45211 non-null  object\n",
      " 10  contact    45211 non-null  object\n",
      " 11  day        45211 non-null  int64 \n",
      " 12  month      45211 non-null  object\n",
      " 13  duration   45211 non-null  int64 \n",
      " 14  campaign   45211 non-null  int64 \n",
      " 15  pdays      45211 non-null  int64 \n",
      " 16  previous   45211 non-null  int64 \n",
      " 17  poutcome   45211 non-null  object\n",
      " 18  response   45211 non-null  object\n",
      "dtypes: int64(8), object(11)\n",
      "memory usage: 6.6+ MB\n"
     ]
    }
   ],
   "source": [
    "bm.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>salary</th>\n",
       "      <th>balance</th>\n",
       "      <th>day</th>\n",
       "      <th>duration</th>\n",
       "      <th>campaign</th>\n",
       "      <th>pdays</th>\n",
       "      <th>previous</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>45211.000000</td>\n",
       "      <td>45211.000000</td>\n",
       "      <td>45211.000000</td>\n",
       "      <td>45211.000000</td>\n",
       "      <td>45211.000000</td>\n",
       "      <td>45211.000000</td>\n",
       "      <td>45211.000000</td>\n",
       "      <td>45211.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>40.936210</td>\n",
       "      <td>57006.171065</td>\n",
       "      <td>1362.272058</td>\n",
       "      <td>15.806419</td>\n",
       "      <td>258.163080</td>\n",
       "      <td>2.763841</td>\n",
       "      <td>40.197828</td>\n",
       "      <td>0.580323</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>10.618762</td>\n",
       "      <td>32085.718415</td>\n",
       "      <td>3044.765829</td>\n",
       "      <td>8.322476</td>\n",
       "      <td>257.527812</td>\n",
       "      <td>3.098021</td>\n",
       "      <td>100.128746</td>\n",
       "      <td>2.303441</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>18.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>-8019.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>33.000000</td>\n",
       "      <td>20000.000000</td>\n",
       "      <td>72.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>103.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>39.000000</td>\n",
       "      <td>60000.000000</td>\n",
       "      <td>448.000000</td>\n",
       "      <td>16.000000</td>\n",
       "      <td>180.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>-1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>48.000000</td>\n",
       "      <td>70000.000000</td>\n",
       "      <td>1428.000000</td>\n",
       "      <td>21.000000</td>\n",
       "      <td>319.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>-1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>95.000000</td>\n",
       "      <td>120000.000000</td>\n",
       "      <td>102127.000000</td>\n",
       "      <td>31.000000</td>\n",
       "      <td>4918.000000</td>\n",
       "      <td>63.000000</td>\n",
       "      <td>871.000000</td>\n",
       "      <td>275.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                age         salary        balance           day      duration  \\\n",
       "count  45211.000000   45211.000000   45211.000000  45211.000000  45211.000000   \n",
       "mean      40.936210   57006.171065    1362.272058     15.806419    258.163080   \n",
       "std       10.618762   32085.718415    3044.765829      8.322476    257.527812   \n",
       "min       18.000000       0.000000   -8019.000000      1.000000      0.000000   \n",
       "25%       33.000000   20000.000000      72.000000      8.000000    103.000000   \n",
       "50%       39.000000   60000.000000     448.000000     16.000000    180.000000   \n",
       "75%       48.000000   70000.000000    1428.000000     21.000000    319.000000   \n",
       "max       95.000000  120000.000000  102127.000000     31.000000   4918.000000   \n",
       "\n",
       "           campaign         pdays      previous  \n",
       "count  45211.000000  45211.000000  45211.000000  \n",
       "mean       2.763841     40.197828      0.580323  \n",
       "std        3.098021    100.128746      2.303441  \n",
       "min        1.000000     -1.000000      0.000000  \n",
       "25%        1.000000     -1.000000      0.000000  \n",
       "50%        2.000000     -1.000000      0.000000  \n",
       "75%        3.000000     -1.000000      0.000000  \n",
       "max       63.000000    871.000000    275.000000  "
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "bm.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count    45211.000000\n",
       "mean        40.197828\n",
       "std        100.128746\n",
       "min         -1.000000\n",
       "25%         -1.000000\n",
       "50%         -1.000000\n",
       "75%         -1.000000\n",
       "max        871.000000\n",
       "Name: pdays, dtype: float64"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "bm.pdays.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count    8257.000000\n",
       "mean      224.577692\n",
       "std       115.344035\n",
       "min         1.000000\n",
       "25%       133.000000\n",
       "50%       194.000000\n",
       "75%       327.000000\n",
       "max       871.000000\n",
       "Name: pdays, dtype: float64"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "bm1=bm.copy()\n",
    "bm1.drop(bm1[bm1['pdays'] < 0].index, inplace = True) \n",
    "bm1.pdays.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x231d80c1648>"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "bm1.groupby(['education'])['balance'].median().plot.barh()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "bm1.pdays.plot.box()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "no     0.769287\n",
       "yes    0.230713\n",
       "Name: response, dtype: float64"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "bm1.response.value_counts(normalize=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    6352\n",
       "1    1905\n",
       "Name: response, dtype: int64"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "bm1.replace({'response': {\"yes\": 1,'no':0}},inplace=True)\n",
    "bm1.response.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Object data type features  ['job', 'marital', 'education', 'targeted', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']\n",
      "Numerical data type features  ['age', 'salary', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous', 'response']\n"
     ]
    }
   ],
   "source": [
    "# here we are seperating object and numerical data types \n",
    "obj_col = []\n",
    "num_col = []\n",
    "for col in bm1.columns:\n",
    "    if bm1[col].dtype=='O':\n",
    "        obj_col.append(col)\n",
    "    else:\n",
    "        num_col.append(col)\n",
    "print(\"Object data type features \",obj_col)\n",
    "print(\"Numerical data type features \",num_col)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 576x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 576x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAfoAAAGFCAYAAAAVYTFdAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjMsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+AADFEAAAgAElEQVR4nOzdd3hb5f3+8fdHlvfIcHacPSADCJABTQhQ9vhCGS2lUEZZhaaFUijQllFKGWEVftCyyt6bAAmBtiQkhCygBBLIINOZznC8JUt+fn9IoY5xEieWdGT5fl2XLvsMnfOxZevWc85znmPOOURERCQ1+bwuQEREROJHQS8iIpLCFPQiIiIpTEEvIiKSwhT0IiIiKUxBLyIiksIU9CKAmd1kZq7eY52ZvWNm+3pdW2tjZgOjr0dbr2tpjJllROsbFsNtjjMzXesscaGgF/mfrcDB0ccVwEDgAzNr72lVrc9A4EYgKYMeyCBSX8yCXiSe/F4XIJJEQs65mdHvZ5rZcuAT4Fjgec+qkmYxs2znXLXXdYh4RS16kR37Ivq1R/2ZZtbezB42s/VmVmNmM8xsVIN1LjCz+WZWbWYbzWyqmQ2JLusdPT3wMzN7xszKzWyDmd3YsAAz+6GZzYruZ72Z/d3M8uotPyy6rcPM7BUzqzCzpWZ2WYPtDDGz98xss5lVmtnXZvarBuucbGZzo/taZ2bjzSx9R78cM/tzdD1fg/knRmvqH50+ycw+je53S/TnOXQH2zwMeDs6uSy6neXRZV3N7PHoz1dtZovM7BYzy6j3/G2/27PM7GkzK922PTPLNLN/mFmpmW0yszvN7IqGh8yb8PqWR78+Ue9UT+/oc7Oiv7dVZhYwsy/M7PgG2880sweidWw2s3uBHf6eRZpLQS+yYz2jX5dtm2FmmcC/gKOAq4EfASXAv8ysS3SdscBDwLPAccAvgBlAmwbbvxOoAk4HHgVurB++ZjYYeA/YCJxG5HDxz4BXG6n1USIfTE4BpgAPmtnIessnAGHgbOAk4P8B+fX29RPgdWB2dPmfgYuB23by+3kR6Aw0DO2fAJ8655aYWb9ovf8B/g84C3gH2NHpkM+Aq6Lfn0rkNMop0ekOwGbgSiJHWe4Ezo/+LA3dRSSQfwzcGp03Hjgv+rOdReT1/V39JzXl9QV+GP16C/871bM2Ou/V6D5ujf68c4AJDc7n3w5cCPwlWkevhnWIxJRzTg89Wv0DuIlIoPqjj37AB8DnQGa99S4AgsCAevP8wLfAndHpq4gE3Y721RtwwPsN5j8KrAZ80ekXgcVAWr11fhJ97sHR6cOi0zfXWyedSDjdHp3uEF1nnx3UY8AK4IkG838BVAOFO/lZvgAeqjedSaSvw1XR6dOBTbv5WpwYrbf3LtbzE/ngUwNkNPjdvtFg3cLoz3J1g597fuRtcLde37zoPs5rsI8jovMPbTD/I+CVBnVcU2+5D/imfh166BHLh1r0Iv9TCNRGH0uA/YFTnXOBeuscCXxK5LCy38y29XOZCgyPfv9fYH8zu9fMxtY/tNzAGw2mXwe6AUXR6ZFEAitcb53XgBAwpsFz39/2jXOulsgHhG3b2QysAh4yszPMrFOD5w4k0rp9edvPFP25/gNkAUN3UD/AS8Bp9X4PxxE5UvBydPpLoI2ZPWVmR5tZ7k62tVMWcYWZLTCzaiKv03NEPlz0bLD6uw2m94n+LBO2zXDOOf53mmCbpry+O3IksA74uMHv8d/1nrutjrfq1VFXf1ok1hT0Iv+zFRgBHARcQqR39fMNzkF3iC6vbfA4n+i5fOfcv6LTY4kcRt8YPbfeMOQ27GC6a72v6+uvEA39TXz/0Hdpg+kgkUDZFiRHEwmhx4F1ZjbNzPav9zMBTGzwM207ZbFdH4UGXow+f9vh7DOAT5xzK6P7XgicDPSNbn+jmT1vZh13ss0duQK4m8gHpJOJfBDadqojq8G66xtMbzvsXtJgfsPpXb6+O9Ehup+Gz72p3nO31bGj114k5tTrXuR/Qs65udHvZ0VbjU8TOc/7UnT+ZmAucGkjz/+u5e+cewp4KhpopwL3AmXAtfXWb9iy3ja9tt7X7dYxszQiRx42N/3HAufcN0Ra3unAIcAdwLtmVlRvWxcTOVXR0LJG5m3b7lIzmwucYWbTiZyX/kODdd6N7qsNcALwNyLn1X+6Oz8DkdfhFefcH7fNiPZjaLS0BtProl87sv3vruEHjia9vjuwmciplx/tZJ1tdXRqUEfDvwWRmFHQi+zYs8A10ce2oP83kdbxSufcLlthzrkS4GEzOxVoGEqnAP+oN30qkXAvjk7PAk4xsz/UO3x/KpH/2+m7/+N8d1j/P2Z2D5FLBtsCC4kEVG/n3KN7sNkXgT8SOdSfDbyyg31vJXKE5FAiHdh2JBj92rCVns33w/asJtb4JZFz+ScT6ZSHmRmRDyb1NeX13VF9/ybSqa4i+sFqV3V8E63DF50WiQsFvcgOOOecmd0KPGdmRzjn/k2khf9LYIqZ3QUsJdLCHgmsc87da2Z/JnJofQqRDn77E+mZfm2DXQwxs4eJnHcfS6Qj2OXRQ+0Q6dX9OfCmmf2DyDn3O4DJzrlPmvpzWGR0v7uIfFhZCrQj8uHlC+fc5ug6vwOeMbMCYBKRMOtLpHV6unOuaie7eJlID/g7gY+cc9uOSGBmlxAJ9feANcAAIi3zp3eyvYXRr5eY2YtAlXPuSyKdI39jZrOIdI47C+jflN+Bc26TmT0K/NnMaoGviRyOL2D71v8uX1/nXNDMlgE/MbOviAT3vGh9k4kMsnQHkY5+BUQG1slyzl0XreORaB2h6DoXEengJxIfXvcG1EOPZHgQ7XXfyPw0YBGRcN02rw1wH5EObkEiLfDXgdHR5ScSad2VEAmBhURC3qLLexMJl7OAF4hcBlZC5LIva7D/I4i07GuInMf9O5BXb/lh0W0NbfC8KcCr0e87Ac8QCa0aIoePXwB6NnjOccA0oJLIaYb/Evmw4W/C7296tI5LGsw/mEjHuDXRfS8j8mElcxfb+x2RKwFCwPLovDzgCSKHvDcDj/G/HvpDG/xuT2xkm1lEjqBsBbYA90df99IG6+309Y2uczSRcK+h3hUCRDoG/plIZ85g9Hf9HnBCvedmRl/HbXX8PyKXDDqv/w/0SM3HtjceEUmQ6OAqy4D/c8694201rZuZ/QtId841OoCPSCrQoXsRaRXM7HBgFJFBedKJXCFwBJFTCSIpS0EvIq1FBZE+B9cROYy/mMigN42NNCiSMnToXkREJIVpwBwREZEUpqAXERFJYSl3jr5Dhw6ud+/eXpchIiKSMJ9++ulG51yjQ0unXND37t2buXPn7npFERGRFGFmK3a0TIfuRUREUpiCXkREJIUp6EVERFKYgl5ERCSFKehFRERSmIJeREQkhSnoRUREUpiCXkREJIUp6EVERFKYgl5ERCSFKehFRERSmKdBb2aPm9kGM/tqB8vPMrN50ccMM9sv0TWKiIi0ZF7f1OZJ4AHg6R0sXwYc6pzbYmbHAY8AoxJUm4hIyqqoqODPf76J8vLyRpefdNLJHH/88QmuSuLB06B3zn1kZr13snxGvcmZQFG8axIRaQ2mT5/OnDlzGdS2lnSf225ZcWUazzy1heOOOw4z86hCiRWvW/S74wJgktdFiIikgmnTplGYDdfuX0bDLJ+yJpPHv9nAt99+S//+/b0pUGKmRXTGM7PDiQT9NTtYfrGZzTWzuSUlJYktTkSkhamqqmLO7NkcWFj9vZAHOKBDEDP46KOPEl+cxFzSB72Z7Qs8BpzsnNvU2DrOuUecc8Odc8M7duyY2AJFRFqYqVOnEqytZWSnYKPLCzIcg9qF+GDye9TV1SW4Oom1pA56M+sJvA783Dm3yOt6RERSwaRJE+mS6xjQJrTDdcZ2qWbt+g188cUXCaxM4sHry+teAD4B9jKzYjO7wMx+aWa/jK5yA1AI/N3M/mtmcz0rVkQkBRQXFzNv3pcc0rmq0cP22wzvGCQnHd59993EFSdx4XWv+zN3sfxC4MIElSMikvJee+010nxwSNfATtfLSIMfdK5myocfcumll1JYWJigCiXWkvrQvYiIxE55eTkTJ77LwZ1qaJvpdrn+MUU1hMNh3njjjQRUJ/GioBcRaSUmTJhAIBDk2J41TVq/c04dB3QM8Nabb1BdXR3n6iReFPQiIq1AVVUVL734AvsW1tIzL9zk553Qs4byikrefPPNOFYn8aSgFxFpBV5//XXKyis4pU/Vbj2vf5sQ+xbW8sLzz1FVtXvPleSgoBcRSXHl5eW89OILDOsQpF/Bji+p25FT+1RSVl7Bq6++GofqJN4U9CIiKe7ZZ5+loqKS03azNb9N34IwB3YI8sLzz7FpU6PjlkkSa0lj3UsKCgQCfP755zscfWvfffclLy8vwVWJpI7Vq1fz+muvMqZrDb3ym35uvqEz+ldy3ewMHn/8ca6++uoYVijxpqAXTz366KM7PRx4+OGHc+ONNyawIpHU8tBDD+GjjtP7Nu/8epecOo7sXs2kiRM55ZRTdLObFkSH7sUzGzdu5M0336K2fR8qB5/0vUew0yA+nDKFZcuWeV2qSIs0a9Yspk2bxv/1rKRdE66b35Uf9a4mL8Nx7z33aAz8FkRBL5557rnnCIXDBLofSF1uh+89At0PwNLSeeKJJ70uVaTFCQQC3Pe3e+ma6ziuZ2yugc9Nd5zRt4L5Cxbw3nvvxWSbEn8KevHE4sWLeeuttwh2GIDLKmh8JX8mNZ2G8NFHU5k7V7c5ENkdzz//PGvWruOcAWWkx/CdfkyXAHu1DfHQP/5OaWlp7DYscaOgl4QLhULcMX48zp9FoGj4TtcNdt0Hsttw5113UVPTtNG8RFq75cuX89yzz3JQ5wBD2u/+5XQ7YwbnDqygsqKCBx98MKbblvhQ0EvCvfzyyyxZvJiqHiPBn7nzlX1+qnr+gPXr1vHoo48mpkCRFqyuro47x48nKy3M2QMq47KPorwwJ/Sq4oMPPmDOnDlx2YfEjoJeEmrevHk89thj1LbrTahdnyY9J1zQlWCnwbz22mt89NFHca5QpGV7++23mb9gAWf2q6Ago/kd8HbkpF7VdM113HPXnRoHP8kp6CVhNm/ezA033kRdZj41fcaw05thNxDoMYK6vI7cdtvtFBcXx7FKkZZrw4YNPPzQPxjSvpYxXXZ+G9rmykiD8weWsXb9Bp544om47kuaR0EvCREIBPjTn65na1kZlX0Ph7SM3duAL42qvodTE6rjuuv+QHl5eXwKFWmhnItc9hYOBjh/r4rd+Ry9x/ZuF+LwbjW8+sorfPPNN/HfoewRBb3EXTgc5pZbbmHBgvlU9R5LXU77PdqOy8yjsu/hFK9ezR/++EeCwWCMKxVpuT788EM+mTmTU/tU0Ck7cde4n9G/ijaZjvG3304oFNuOfxIbCnqJK+ccDz74INOmTaOmxyhC7Xs3a3vhgq5U9R7Dl/Pmcdttt2nQDhEiN625/76/0acgzNFFib06JcfvOGdAGUuXL+ell15K6L6laRT0EldPPvkkr7/+OsHOQ6jtMiQm2wwV9qOmaAQffvgh99xzD87Fr8ORSEvw8MMPs7WsjPP3KifNg3f1AzvWcmCHIE89+SRr1qxJfAGyUwp6iZvnnnuOp556imCHAQR6jIzptmu77kOg67688847PPDAAwp7abW++uor3nnnHY4pqqZ3M25a01w/H1hJmgtx77368J1sFPQSF6+++iqPPvoote37Eug9erd62DdVsPuBBDsP4bXXXuPRRx/Vm4u0OuFwmPv+9jfaZcGpe3gL2lhpn1XHKX0qmDNnLjNmzPC0Ftmegl5i7o033uCBBx4g1K4XNX3GgsXpz8yMQI+RBDvuzfPPP69LfKTVmTRpEouXLOGnfcvJSoJ7kR7ZvYbueXU88P/uJxCI7+V90nQKeompt956i/vuu49Q255U9z0MfHH+EzMj0Otggh0G8vTTT/Pkk0/Gd38iSaKyspLHHnmYgW1DHNQ5Oa5A8fvgrP4VrF23ntdee83rciRKQS8xM2nSJO69915CbXtQ3e9w8KUlZsdmBHqPprbDAJ588kmee+65xOxXxEOvvvoqpWXl/Kx/Yq6Zb6qh7WvZrzDIC889R0VFhdflCAp6iZEPP/yQ8ePHE27Tnep+P0xcyG9jRk3v0dS278ujjz7KG2+8kdj9iyRQWVkZL734Igd2CNK3wLsOeDtyWt8qyisreeWVV7wuRVDQSwzMnDmTv/zlFsJ5najyIuS3MR81fcYSatuT++67T/fLlpT1yiuvUFVdzSl9ve2AtyO988MM7xjglZdf0iiWSUBBL82yePFibrjhRsLZ7ajsfxSkpXtbkM9Hdb/DCBd0Y/z48Xz22Wfe1iMSY8FgkAlvvcn+HYL0zEu+1vw2J/Wupqq6Rh+4k4CCXvbYpk2buPa6PxC0dCoHHAn+3Ry/Pl58fqr6/5BwVhuuv/4G3QRHUsrUqVPZWlbOUQkeAW939c4P079NmDffeEMjWHpMQS97JBQK8afrr2fzllIq+x2BS8/xuqTtpWVQ2e8IqoIhrrn2WqqqkvMQp8jueuvNN+iS6xjcrtbrUnbpiO5VrF6zhs8//9zrUlo1Bb3skZdeeomvFyygqtdo6nILvS6nUS6rgMq+h7O6uJh//vOfXpcj0mybNm3iq/kLGNO5Gl8S9bTfkREdg2T6jWnTpnldSqumoJfdtnLlSp544klq2/UmVNjX63J2KlzQlWCnQbz2+uvMnz/f63JEmmXWrFkADOuQHNfN70pGGgxtG2DGx9M1cqWHFPSy2+6///8Rxkeg10Fel9IkgaLhkJHLXXffrTcbadE++eQTCrOhR27ydsJraFiHIBtKNrJs2TKvS2m1FPSyW4qLi5k7dw41nYck33n5HUlLp7rrMJYtXcqXX37pdTUie2zhN18zoCCQVAPk7MrAtpG+BAsXLvS4ktbL06A3s8fNbIOZfbWD5WZm95vZEjObZ2YHJLpG2d67774LZtR2GOh1Kbsl1L4P5s/gnXfe8boUkT1SVVXFhpKNFLWg1jxAp6w6/L7IKT/xhtct+ieBY3ey/DhgQPRxMfCPBNQkO/HhlKmECopwGS2kNb9NWjqBtr2ZOvUjHb6XFmlbUHbNiW3QP7soh1s/K/ju8eyi2P5vp/mga24dy5cvj+l2pek8DXrn3EfA5p2scjLwtIuYCbQ1s66JqU4aqquro2TDesLZbb0uZY/UZbcjEKihrKzM61JEdtu2EebaZMT2mvSVFX6+KU3/7rGyIva3wcv3hygr2xrz7UrTeN2i35XuwKp608XRedsxs4vNbK6ZzS0pKUlYca1NaWkp4XAYl5HrdSl7ZNtRCP2NSEsUCoWASAu5pfHb/+qXxEv2P5nGupx877irc+4R59xw59zwjh07JqCs1qnlj24V+XMKh1vWOU4RqBf01vJOPaX5IKyg90yyB30x0KPedBGwxqNaWr3CwkLy8vPxVe3sbEvy8lVvxszo3bu316WI7La8vDwAKmtbUJf7qIpaH3n5+V6X0Wole9BPAM6J9r4/CNjqnFvrdVGtlZkxcMAA0qs2el3KHkmrLKGoRw8yMzO9LkVkt3Xp0gWAjTUe3R2yGTYG/HTu3MXrMlotry+vewH4BNjLzIrN7AIz+6WZ/TK6ykRgKbAEeBS4zKNSJeoHP/gBVrUZX8UGr0vZLVZThr9sDWNGj/a6FJE90rFjR3xmlFQne/tse7V1UFrzvw8qknix7165G5xzZ+5iuQN+laBypAmOP/54nnjyKUJr51E94Eivy2myjHVfkZaWxumnn+51KSJ7xO/306dPb5ZsWQRUe1xN03271Y8D+vfv73UprVbL+mgonsvJyeH0007FX7qStPJ1XpfTJL7qLWRuWsyxxxxDYWFy3oBHpCkOOHA4i7emE2xB/UkXbEnHZ8awYcO8LqXVUtDLbjvjjDPo3KULOcunQTjJb65RFyZn2Ufk5eVxwQUXeF2NSLMccMAB1NbBoq3pXpfSZF9tyWDAwAHkqzOeZxT0sttycnK44frrsUAFWStmQhKPNJex+jOschPXXvN72rdv73U5Is2y//77k5OdxYx1GV6X0iTrq3ws2epnzJhDvC6lVVPQyx4ZMmQI55xzDumblpC+Pjlv/+rf9C2Z677khBNOYLQ64UkKyMrK4ogjj2JOSTbVoeS/zG7a2kx8ZhxzzDFel9KqKehlj5177rkcMnYsWatm49+83OtytpNWvo7s5dPZd9/9uPzyy70uRyRmjjvuOAJhxyfrk7tVH6qDaetzGDFiBJ06dfK6nFZNQS97zOfz8ac//pFBgweTs+yjpOmc56veQu63/6aoezf++tdbyMhI7jdEkd0xaNAgBg7oz8RVuYSTeLDKGesy2VIDJ//oR16X0uop6KVZMjMzue3WW+natTO5S/6Fr8LbceStZit5iybTJi+XO8ePVwcgSTlmxs/POZcNVcbMDcn5IbbOwdsrc+nfvx8HH3yw1+W0egp6aba2bdvyt3vvpWNhe/KWvI+vapMndVignLxFk8nLSue+v91L16660aGkptGjR9O3T28mrMhLylb9J+szWF9lnHPOuZglf1+CVKegl5jo1KkT9/3tXtoX5JO3aDK+qi0J3b8FKshb9B65frj3nrvp1atXQvcvkkg+n49fXHAhayuND9ck15DOgTC8uiyfAf37M2bMGK/LERT0EkNdu3blvvv+Rpu8HPIWv4evOjFhvy3ks31h7r77Lo3AJa3C6NGjGbbffry+PC+pbnTz3spsNlXDr8aNw+dTxCQDvQoSU0VFRdx/398oyMmMtOyrS+O6PwtWkrd4MtkW4p6772bvvfeO6/5EkoWZ8atx46ishTeXZ3tdDgCbAz7eWZXDIWPGaCS8JKKgl5jr2bMn9993H/nZGeQunowFyuOyH6utJm/RZDJdgLvuupNBgwbFZT8iyWrAgAEcf/wJfFCczcpy7+9q99yiHJylc+lluv9YMlHQS1z06tWLe++5mxw/5C2ajAWrYruDUJDcxe+THqpk/B13MGTIkNhuX6SFuOSSS8jPz+eJRfnUeThI5Rcb05lTksnPzzmHbt26eVeIfI+CXuKmX79+3HXnnWS6ILmLJ0MoEJsN14XIXfIB/ppSbrnlFvbbb7/YbFekBSooKOCyX43j261pTPGoY14gDE8vyadnjyLOOOMMT2qQHVPQS1wNHjyY22+/DX+gjJxv/wN1zbztlnNkLf0IX8UGrr/+ekaNGhWbQkVasKOPPpr9hw3jpaV5bA4k/m399aU5lFQZV/7uKg1QlYQU9BJ3+++/P9dccw1pZWvJXDGjWTfByVj9KelblvPLSy7hsMMOi12RIi2YmfG7q64iTDpPL8xN6H2mlpal8V5xNieeeKI64CUpBb0kxNFHH825555LxsbFe3wTHP+mb8lcO48TTjhBhwdFGigqKuL8X/yCzzZmMKckMa3qUB08vrCA9m3bcskllyRkn7L7FPSSMOeddx5jxowha/Xc3R4q12rKyFkxgyFDhvLb3/5Wo22JNOLHP/4x/fv349nF+Qm5tn7SyixWlvu44srfabjpJKagl4QxM6655ho6FBaSu2wqhINNe2JdmJylU8jJyuSGG67H7/fHt1CRFsrv9/P7319DWa2PF5fkxHVf66p8vLk8l7FjD+GQQ3S/+WSmoJeEys/P58YbbsAXrCBz1ZwmPSdj7Tx8lRu59tpr6Ny5c5wrFGnZBg4cyBlnnMHUtVl8vSU+H4qdgycW5pORlc3ll18Rl31I7CjoJeH22WcfTj31VDJKFuKr3LjTda2mjMx18zj88MPVahBponPPPZeuXTrz5KICauNw05vp6zL5eoufX156GYWFhbHfgcSUgl48cd5559G2bTuyV36y0174WatmkZmezmUaaUukybKysrjit1eyttKYuCK2w+OW1xovfpvHkMGDOeGEE2K6bYkPBb14Ii8vj1/+8hJ8FSX4S1c2uo6vYgP+0lWcc87P6dixY4IrFGnZRo0axWGHHcaEFTmsr4rdW/0r3+ZQFfbxu6uu0k1rWgi9SuKZI488ks5dupC57stGW/WZa+eRm5vHKaec4kF1Ii3fuHHjSEvP4MVvc2OyveXlaUxdk8Vpp51O3759Y7JNiT8FvXjG7/fzszPPxFexgbTyddst81WX4i9dyemnn0ZOTnx7D4ukqg4dOvDzc87l05IM5m9uXsc85+DZxXm0KcjnnHPOiVGFkggKevHUscceS3Z2Dumblmw3379xCT6fj5NPPtmjykRSw+mnn07Xzp14bknzbnoze0MGi0r9XHjxJeTl5cWuQIk7Bb14KjMzk8MOO5SM0hVQF4rMdI7M0mUceOBw2rdv722BIi1cZmYml1x6GcUVPj5Zt2cj5oXq4LXlefTt3ZvjjjsuxhVKvCnoxXNHHHEELhTEv3U1AL7KEqgp56ijjvS4MpHUMHbsWAb078/rK/II7cHldtPWZrKu0rjw4otJS/P+vveyexT04rn99tuPjIxM0srWAuCPfh05cqSXZYmkDJ/Px4UXXURJlTF1N29lW1sHb63IY/DgQRx88MFxqlDiSUEvnktPT2ffffchvSLSIc9fvpbevfvQtm1bjysTSR0jR45k8KC9mVicS3g3WvUz1mWyuQbOP/8XusdEC6Wgl6QwbNgwrGozhAL4K0sYNmw/r0sSSSlmxpk/O4uSKmvy3e3qHExclUv//v0YPnx4nCuUeFHQS1Lo168fAP7SVbhwLf379/e4IpHUM3r0aHoUdWfiqqbds/6/G9NZW2mceebP1JpvwRT0khS2Db6Rvunb7aZFJHZ8Ph+n//gnLC/zsbR819fV/2dNNh0K23PooYcmoDqJF0+D3syONbOFZrbEzK5tZHlPM/vQzD43s3lmdrwXdUr8derUifSMDPxlkZ73PXr08LgikdR05JFHkpmZwZTVO++UV1Lt48tN6Zxw4v/p1tAtnGdBb2ZpwIPAccBg4EwzG9xgtT8BLzvn9gd+Cvw9sVVKopgZHTt2AiArO4f8/HyPKxJJTbm5uRxxxJHM3JBFdWjH6wtce6IAACAASURBVH20NhMz4/jj1b5q6bxs0Y8EljjnljrngsCLQMNh0BxQEP2+DbAmgfVJgnXt0gWATp10AxuReDr22GMJhOHzjY13ynMOZm7IZtj+w+jcuXOCq5NY8zLouwOr6k0XR+fVdxNwtpkVAxOBXze2ITO72MzmmtnckpKSeNQqCdCuXeRyukKNhicSV0OHDqWwfTtmb2j88P2KijTWVxk//OERCa5M4sHLoG+sC2fDfqBnAk8654qA44FnzOx7NTvnHnHODXfODdftTFuuNm3aAOj6eZE48/l8HHb4D5m3OaPR8e/nbMjA5/MxZsyYxBcnMedl0BcD9XtcFfH9Q/MXAC8DOOc+AbKADgmpThKuoCBylkZ3qxOJvzFjxhCqg8rQ99tc8zZnMXToEH3oThFeBv0cYICZ9TGzDCKd7SY0WGclcASAmQ0iEvQ6Np+isrOzgUhrQ0Tia8iQIWRmZlBZu/3/W6gOVpT7GDlylEeVSax59o7qnAsB44DJwNdEetfPN7Obzeyk6Gq/Ay4ysy+AF4DznGvKMA/SEmVm7t4Y3CKy5zIyMth//wO+16LfNj1ixAgvypI48PTiSOfcRCKd7OrPu6He9wuA0YmuS0SkNRg2bBgzZ87cbl51yEdOdhYDBgzwqCqJNR0jFRFppQYPbjh0CVSHjUGDB+sUWgrRKyki0krttdde35sXCBuDBw/xoBqJFwW9iEgrlZmZSVYjfWMGDhzoQTUSLwp6EZFWLCt6tUt9ffr08aASiRcFvYhIK5bdIOjNjK5du3pUjcSDgl5EpBVreFlrZmYGaWlpHlUj8aCgFxFpxTIyMhpMazyLVKOgFxFpxb7folfQpxoFvSQNs8bucyQi8dTwevn09HSPKpF4UdBL0tEoxyLeUdCnHgW9JB217EW8o6BPPQp6SRpqyYt4z+/39BYoEgcKehER+Y4urUs9CnpJGjpkL+I9BX3qUdCLiMh39IE79SjoJenoXL2ISOwo6CXpqEUhIhI7CnoREZEUpqAXERFJYQp6ERGRFKagl6SjzngiIrGjoJekoU54IiKxp6AXERFJYQp6ERGRFKagFxERSWEKehERkRSmoBcREUlhCnoREZEUpqAXERFJYQp6ERGRFKagFxERSWEKehERkRTmadCb2bFmttDMlpjZtTtY5ydmtsDM5pvZ84muUUREpCXze7VjM0sDHgSOAoqBOWY2wTm3oN46A4DrgNHOuS1m1smbakVERFomL1v0I4Elzrmlzrkg8CJwcoN1LgIedM5tAXDObUhwjSIiIi1ak4LezHLM7HozezQ6PcDMTmzmvrsDq+pNF0fn1TcQGGhmH5vZTDM7tpn7FBERaVWa2qJ/AggAB0eni4Fbmrnvxu5J2vBG5H5gAHAYcCbwmJm1/d6GzC42s7lmNrekpKSZZYmIiKSOpgZ9P+fceKAWwDlXTeNBvTuKgR71pouANY2s85ZzrtY5twxYSCT4t+Oce8Q5N9w5N7xjx47NLEtERCR1NDXog2aWTbTFbWb9iLTwm2MOMMDM+phZBvBTYEKDdd4EDo/uswORQ/lLm7lfERGRVqOpve5vBN4DepjZc8Bo4Lzm7Ng5FzKzccBkIA143Dk338xuBuY65yZElx1tZguAMHC1c25Tc/YrIiLSmjQp6J1zH5jZZ8BBRA7ZX+6c29jcnTvnJgITG8y7od73Drgy+hAREZHd1NRe96OBGufcu0Bb4A9m1iuulYmIiEizNfUc/T+AKjPbD7gaWAE8HbeqREREJCaaGvSh6GH0k4H7nXP3AfnxK0tERERioamd8crN7DrgbGBsdPja9PiVJSIiIrHQ1Bb9GUQup7vAObeOyAh2d8atKhEREYmJpva6XwfcU296JTpHLyIikvSa2uv+VDNbbGZbzazMzMrNrCzexYmIiEjzNPUc/Xjg/5xzX8ezGBEREYmtpp6jX6+QFxERaXma2qKfa2YvERl7/rsx7p1zr8elKhEREYmJpgZ9AVAFHF1vngMU9CIiIkmsqb3uz493ISIiIhJ7Te11X2Rmb5jZBjNbb2avmVlRvIsTERGR5mlqZ7wniNwrvhuRwXLejs4TERGRJNbUoO/onHvCOReKPp4EOsaxLhEREYmBpgb9RjM728zSoo+zgU3xLExERESar6lB/wvgJ8C66OP06DwRERFJYk3tdb8SOCnOtYiIiEiMNbXXfV8ze9vMSqI9798ys77xLk5ERESap6mH7p8HXga6Eul5/wrwQryKEhERkdhoatCbc+6Zer3unyUyMp6IiIgksaYOgfuhmV0LvEgk4M8A3jWz9gDOuc1xqk9ERESaoalBf0b06yUN5v+CSPDrfL2IiEgSamqv+z7xLkRERERir6m97n9sZvnR7/9kZq+b2f7xLU1ERESaq6md8a53zpWb2RjgGOAp4KH4lSUiIiKx0NSgD0e/ngD8wzn3FpARn5JEREQkVpoa9KvN7GEiw+BONLPM3XiuiIiIeKSpYf0TYDJwrHOuFGgPXB23qkRERCQmmhT0zrkqYAMwJjorBCyOV1EiIiISG03tdX8jcA1wXXRWOvBsvIoSERGR2GjqoftTiNy9rhLAObcGyI9XUSIiIhIbTQ36oHPOER3f3sxy41eSiIiIxEpTg/7laK/7tmZ2EfAv4LHm7tzMjjWzhWa2JDqW/o7WO93MnJkNb+4+RUREWpOmDoF7l5kdBZQBewE3OOc+aM6OzSwNeBA4CigG5pjZBOfcggbr5QO/AWY1Z38iIiKtUZOvhXfOfeCcu9o5dxXwHzM7q5n7Hgkscc4tdc4FidwZ7+RG1vsLMB6oaeb+REREWp2dBr2ZFZjZdWb2gJkdbRHjgKVErq1vju7AqnrTxdF59fe/P9DDOfdOM/clIiLSKu3q0P0zwBbgE+BCIoPkZAAnO+f+28x9WyPz3HcLzXzAvcB5u9yQ2cXAxQA9e/ZsZlkiIiKpY1dB39c5tw+AmT0GbAR6OufKY7DvYqBHvekiYE296XxgKDDFzAC6ABPM7CTn3Nz6G3LOPQI8AjB8+HCHiIiIALs+R1+77RvnXBhYFqOQB5gDDDCzPmaWAfwUmFBvf1udcx2cc72dc72BmcD3Ql5ERER2bFct+v3MrCz6vQHZ0WkDnHOuYE937JwLRc/3TwbSgMedc/PN7GZgrnNuws63ICIiIruy06B3zqXFc+fOuYnAxAbzbtjBuofFsxYREZFUpFvNioiIpDAFvYiISApT0IuIiKQwBb2IiEgKU9CLiIikMAW9iIhIClPQi4iIpDAFvYiISApT0EvScE63KRARiTUFvSSd6E2MREQkBhT0IiIiKUxBLyIiksIU9CIi8p1wOOx1CRJjCnpJOuqUJ+KdUCjkdQkSYwp6SRrbAl6d8US8o6BPPQp6ERH5Tm1trdclSIwp6EVE5DvBYNDrEiTGFPQiIq1Yw0P1CvrUo6AXEWnFampqdjotLZ+CXkSkFVPQpz4FvSQN9bYXSbyGwR4Oh9myZYtH1Ug8KOgl6eg6epHEqays/N68r7/+2oNKJF4U9JJ01LIXSYyqqqpGD9XPnz/fg2okXhT0kjS2teTVohdJjAULFnxvXlaaY94XX3hQjcSLgl6ShlryIok1a9YsGv7X5frrmL9gAWVlZZ7UJLGnoJeko8AXiT/nHDM+nk6Ov267+Xnpjrq6OmbPnu1RZRJrCnpJOjp0LxJ/y5cvZ/WateSlb///luV3FGTCtGnTPKpMYk1BL0lj2whdatGLxN+kSZNIM8hP375Fb8BBHauZ8fF0SktLvSlOYkpBL0lj22U+atGLxFcwGGTye5M4oEMAfyMpMLZbgNpQmA8++CDxxUnMKeglaVRUVAAQCAQ8rkQktU2dOpWtZeWM7db4/1rPvDB9C8K89eYbhMPhBFcnsaagl6SxefPm7b6KSOyFw2GeefopivLq2Kf9jm9Je3zPKopXr2HKlCmJK07iQkEvSaOkZCMA6zeUeFyJSOqaMmUKK1cV86Pelfh20h1meMcg3fPqePrJJ9Sqb+EU9JI01qxdC8CG9eupq6vbxdoisruCwSCP//OfdM+rY3jHnd+O1mdwcq9KVqwq5v33309QhRIPnga9mR1rZgvNbImZXdvI8ivNbIGZzTOzf5tZLy/qlPirrq5mw/p11GXkEgwGWL9+vdcliaScF154gdVr1nBmv4qdtua3GdkpSP82YR76x9/ZunVr/AuUuPAs6M0sDXgQOA4YDJxpZoMbrPY5MNw5ty/wKjA+sVVKoixfvhyA2vZ9AVi6dKmH1YiknuLiYp599hlGdQqwb+GOz83X5zM4b69yysvLefjhh+NcocSLly36kcAS59xS51wQeBE4uf4KzrkPnXNV0cmZQFGCa5QE+SI6tnZtx4HgS/tuWkSaLxQKccftt+EnxFkDvn+3up3pmRfm2B7VTJw4UaPltVBeBn13YFW96eLovB25AJjU2AIzu9jM5prZ3JISdeRqiWbNno3LaYfLakM4rzMzZ83yuiSRlPH444/z5VfzOXdAOW0zd3+cilP6VNEjr46/3vIXNmzYEIcKJZ68DPrGzhA1+hdoZmcDw4E7G1vunHvEOTfcOTe8Y8eOMSxREqG8vJx58+ZRmx/5nFfbpjsrV6xgzZo1Hlcm0vJ98sknPP/88xzerYYfdNl5B7wdyUyDcUO2Eqiq4OY/3/TdKJbSMngZ9MVAj3rTRcD33tnN7Ejgj8BJzjmNpJKC3n33XcKhELUd+gEQat8XzMebb77pcWUiLdvy5cu59Za/0Cu/brcP2TfUNbeO8weW8dX8Bdx///0awbIF8TLo5wADzKyPmWUAPwUm1F/BzPYHHiYS8jpelIJCoRCvvPoa4YKu1OUUAuAycqlt15u3336HqqqqXWxBRBqzYcMGrr7qd6SFqvjN0K1kpDV/mwd3CXJir2omTJjAM8880/wNSkJ4FvTOuRAwDpgMfA287Jybb2Y3m9lJ0dXuBPKAV8zsv2Y2YQebkxZqwoQJbNpYQqDTkO3mBzsPobq6imeffdajykRarvLycn5/9VVUbN3MVfuW0jE7duNS/LhvFaO71PD444/zzjvvxGy7Ej9+L3funJsITGww74Z63x+Z8KIkYdasWcNDDz1MuE0R4bY9tltWl9eR2g4DePHFFzn00EPZa6+9PKpSpGUpLy/n6quvonjVSq7abys982M7qp0ZXLB3JeW1adxz991kZGRw9NFHx3QfElsaGU88EQqFuOOOO6itc1T3+kHk3aOBmh4jcf4sbrvtdmpqajyoUqRl2bJlC1dc/huWLFrIuCFlDG4Xn05zfh/8emgZe7et5bbbbuXtt9+Oy34kNhT0knB1dXXcfvvtfPHFF1T1GIXLzGt8RX8mlb1Gs3z5Mv50/fUEg3vWY1ikNSgpKeHy3/yaVSuW8dt9yjigY9MGxdlTmWlw5b5b2bd9LXfffTevvPJKXPcne05BLwnlnOPee+/lX//6F4HuBxLqMGCn64fb9qC69xjmzpnDzTffrMt6RBqxdOlSxv3qMjasLeaqfbeyTxNHvmuujDS4fJ8yRnQM8OCDD/LQQw/pPhVJSEEvCRMIBBg/fjxvv/02gS77Euy2X5OeF+o4kJqeo5g+fTo33njjd/etFxGYNWsW4351GYGyEq4bVsrecTpcvyN+H1w2pIIfdq/hxRdf5Ibrr6e6ujqhNcjOKeglIYqLi7n0ssuYNGkSga77ESw6cLeeX9t5CDU9R/HxjBlcdNHFLF68OE6VirQcr7/+Otddey0d/VXcdMAW+hR4czvZNB+cO7CSswdU8vGMj/nNr8ehUUqTh4Je4m7KlClceNFFLFu5mqoBR0VCvpHOd7tS23kIVXsdz9rNW7n00st4++23NWiHtEqBQIA77riD+++/n/0KA/xx/y20z/L2kLkZHN2jht/uU8aq5d9yycUX6Z4VSUJBL3GzatUqrrvuOm666Saq0vIpH3TS9y6j213h/M5UDDqJQG4n7r77bi6//Aq17qVVKS4u5rJLf8mkSZM4qVcVl+9TTpanF0pvb1iHWm44oJSM4BZ++9vf8sILL+gDuceS6M9DUkV5eTlPP/00r732Os6XRqBoOMHOQ8AXg6G5AJeeTdWAo0kvWciXX3/GRRdfzHHHHsuFF15IYWFhTPYhkoymTZvGbbf+FV+4ht/tV8Z+Cep0t7uK8sL8+cAtPPZ1Lg8//DBfffUl1157Hfn5+V6X1iop6CVmqqureffdd3nyqaeoqKgg2GEgwe4H4NKzY78zM2o77U1t+z5krv2CSZMn858PP+Ssn/2MU045RW8oklICgQAPPfQQb7zxBn0Kwvx6WBkdYjjaXTxk+x3jhlbwfnGIF2fM4KILL+D6G25kyJAhu36yxJSl2iGV4cOHu7lz53pdRqtSUlLCG2+8wVtvTaCysoJwQVdqeoz8buz6RLCaMrKKZ+PfspLMzCxOOOF4TjvtNLp339mdj0WS39KlS7nl5ptZunw5x/So5if9qkiP4UnXWz8r4JvS9O+m925byx8OKIvdDoAlW/384+sCNtWkcd5553HWWWeRlhabI3wSYWafOueGN7pMQS97avHixbz88sv8+z//oa6ujtq2vQh2GUpdXifPavJVbSJj3XzStyzFnGP06DGcccZPGDp0KLYHHQBFvOKc46233uLvDz5Ilq+Wi/aOz6H6RAQ9QHXIeGphLjPWZ7LP0CH88U/X06VLl5jvp7XaWdDr0L3slsrKSj766CMmTprEl/PmYWnpBDrsRbDzEFym94fL63IKqek7lkBwOOkbvubjWXOYPn0aAwYO5ITjj+fwww+nTZs2XpcpslObNm1i/Pg7mDVrNvsW1nLRoHLaZLTsRlm23/HLIRXsUxjk6W/mc8EvzufyK37LUUcdpQ/hcaYWvexSKBTi008/ZfLkyUybPp3aYBCyC6gp3IvajgPBn+l1iTsWriV90xIySxZiVZtJS0vjoIMO5thjj2HUqFFkZGR4XaHIdqZOncrdd91JdVUFZ/St5MiiGnxxzMFEtejr21Dt4+Gv81lc6ufQQw/lyiuv1AfwZlKLXnabc45vv/2W999/n/ff/4DS0i1YeiaBtn2p7dCfutyOe3QtfMKlpVPbaRC1nQbhq9pE+sYlzJjzGR9/PJ3cvHyOPOKHHH300QwePFitCvFURUUF999/P++//z59CsJcMryMbrnJ3eFuT3XKruOP+2/l3RVZvD5tKl/N+4LfX3sdo0aN8rq0lKQWvXynrq6Or7/+mmnTpjF16kesXbsGfD5qC4oIdehPqE2PmF0i5ylXR1rZGtI3LiGjdCWuLkRhh44cOvYQDjnkEPbZZx/8fn0GlsT5/PPPue3Wv7Jx40ZO6lXFSb2r8SdolBMvWvT1rShP46GvC1hd4eOkk07i0ksvJTs7DlfqpDh1xpMdqq2t5bPPPmP69OlMmzad0tItYD5C+V0JtetFqF1vXHqW12XGTziIf8sK/FtWkFG2BlcXIjcvnzGjf8CYMWMYMWIEWVkp/POLpwKBAP/85z95+eWX6ZLruGTvMvq1SexY9V4HPUAwDK8tzeG9Vdl069aVP/zxT7oMbzcp6GU7ZWVlzJ07l+nTpzPjk5nUVFdhaekEC7pHwr1NUXKfd4+XcC3+ravxl64gY2sxLhQgPSODkSNGcsghYxg5ciTt27f3ukpJEYsXL+avt/yF5StWckT3Gn7av5JMDw6YJUPQb/P1Fj+PfNOGLTXGWWefzTnnnEN6evqunyg6R9/a1dXVsWjRImbPns0nM2fyzddf45zDMrIJFPQgVNSLcEFX8LXyP4e0dELtexNq35uaujrSKtbh37KCj+f+l48/ng5A//4DOOigUYwaNYpBgwbpEL/strq6Ol5++WUee/RR8tLDXLVfGfsm6Qh3iTaoXYi/jtjMs4tyeOaZZ5g9aybX33AjRUVFXpfWoqlFn6JKS0uZO3cus2bNYuas2ZSXbQWgLq8jtQXdCbUpoi63A5hud7BLzuGr2ox/azH+stWkVawH58jJzWXE8OEcdNBBjBw5UsPvyi5t2rSJ2279K3M//YwDOwb5xd4V5Kd7+x6cTC36+uZsyODxhfnUpWVy+RW/5ZhjjlGH2Z1Qi74VCIVCLFiwIBrus1m0aGGk1Z6eTTC/G6G+wwgXdIvPcLSpzoy63EKCuYUEu+0HoQD+sjUEtxbz0cy5TJ06FYA+ffty0KhRDB8+nKFDh5KZ2QpPf8gOzZgxgztuv43qynLO36uCw7oFWsSFK14Z0SlI34ItPLwgn9tvv53Zs2dx5ZW/Iy8vz+vSWhy16Fso5xyrV69mzpw5zJkzh88+/5ya6upoKDVstevdJG6cw1e9OXJuf2sxaRUbwNWRnpHBsGHDGDF8OMOHD6dPnz5qjbRSoVCIf/7zn7zwwgv0zK/j0sFldM/15r7xjUnWFv02dQ7eWZHN68ty6NKlC3+55a/069fP67KSjlr0KaKsrIzPPvss0mqfPZuSDRsiC7IKCOb3JNytO6GCrq2zI51XzKjLKSSYU0iw674QriWtfC3+rWuY8+Ui5syeDUDbdu0ZNXIEw4cP58ADD1SnvlaitLSUm2/+M5999jmHd6vh7IGVMR2nvjXwGZzUu5pB7Wp5YD5cdukvufr313DkkUd6XVqLoRZ9EguFQixcuJDZs2cza9YsFi6MHo73ZxDM60q4TTdCBd1xWQVelyo7YIEK0srW4C9bTUb5WlxtDQB9+/Vj1MiRjBw5kqFDh6pncQr65ptvuP5Pf6R08ybOHVjB2G4Br0tqVLK36OsrDRgPzi9gYamf0047jUsvvVQdYqN0eV0LsmHDBmbPnh05JD93LlWVlfUOx0eCvS6vozrRtUTO4avaFDnMX7b6u8P8mVlZHLD/AYwaNZIRI0bojnsp4OOPP+bPN91Egb+WXw8ppU9B8hyqb6glBT1AqA5e+jaHyauyGT78QG6++S/k5OR4XZbndOg+idXW1vLf//6X2bNnM3PWLFatXAmAZeYSyO9GuEt3QgXddTg+FZhRl9uBYG6HSKe+cBB/2VqCW1fzyedf8cknMwDo0rUrB40axciRIznggAM0YE8LM2nSJO68805659Vy5b5bKWjhN6NJNn4fnDWgiqLcMI9/+ilX/vYKbr9jPG3btvW6tKSloPdAeXk5s2bNYvr06cycNSvSic6XRjivM7U9RhBuU0RdVlt1okt1aRnR0Qd7EXAOC5Th37qa4q2reevtd3nzzTejA/aMYPTo0fzgBz/Qm1kSc87xwgsv8MgjjzC0fS2/GVpGlt5h4+bQbgHy0uv4+4JF/Hrcr7jr7nvo3Lmz12UlJf0ZJsi6deuYMWMG06ZP54svvqAuHMYycgi06UGoqCfh/G6Qppej1TLDZbWhNqsNtZ0HU10XJq18Hf7SlXw89ws+/vhjzIzBg4dwyCFjGD16ND169PC6aqnn5Zdf5pFHHuGgTgEuHlyRsLHqW7MDO9Zy9b5b+dtXcMVvfs3fH3qYdu3aeV1W0tE5+jhav349kydP5sMpU1i2dGlkZk67yGh07Xq2nDvAibe2DdhTuoKMrauwyk0AFPXowWGHHsqxxx6rkcM8NnXqVG688UZGdAzwq6EVcb2tbKy1tHP0jfl2q59b/9uGAQP35t6/3dcqx7BQZ7wECgQCfPzxx7w7cSKfffopzjnq8jsTbNOTULueuCzdc1maxwIV+EtXkl66krTyteAcQ/fZhxOOP55DDz1UHZMSbP78+fz2iivomVPDtcNKyWhhN3hMhaCHyEh6D3yVzyFjx3LTTTfh87WuQyoK+gRYvHgxEydO5P33P6CysgIy8wgU9qe2wwBcZn7C65HWwYJVpG9aQuamxVC9lcysLI744Q85/vjjGTJkiAbpibOqqirO+fnZ+Ko3ccMBW1pkx7tUCXqASSuzeGFJLuPGjeP000/3upyEUq/7OPrqq6944MEH+ebrr8GXRm3bntR2Hx25SYwugZM4cxk5BLvuS7DLPqRVrKd242ImTf6AiRMn0qt3by679FJGjRrldZkp67nnnmPjps3ccGBZiwz5VHNsjxq+2pzBE4//kyOOOELn66MU9Htow4YNPPTQQ/znP/+BzFxqeh5EbWE/XQYn3jAjnN+FcH4XanoeRPrmZaxY/yXXXHMNI0eNYtyvfkXPnj29rjKlFBcX8/JLLzKmSw39E3wPeWmcGZw9oII/zE7nscce4+qrr/a6pKSgJuduqqmp4YknnuCss8/mwykfEei6H+VDTqW282CFvCSHtHRqOw6kfPCPqCkawZxPP+e8887ngQceoLy83OvqUsbzzz9PGnX8pF+V16VIPV1z6zi6qJqJEyeybt06r8tJCp4GvZkda2YLzWyJmV3byPJMM3spunyWmfVOfJX/45zjhhtv5KmnnqIqtzvlQ08lWHQgpGn4UklCvjRqu+5D+dDTqGnfj1dffZXfXXUVoZBan83lnGPmJzPYr30NbTN1yD7ZjO0WwDnH7Oi9Jlo7z4LezNKAB4HjgMHAmWY2uMFqFwBbnHP9gXuBOxJb5fbeeustZs+aRU2PUdT0PxyXqdslxpKvYgMZa77AV7HB61JSikvPJtBnDNX9DmPRwoU8/fTTXpfU4n377bds3lLKvoW1XpcijeiWE6YwG+bMmeN1KUnByxb9SGCJc26pcy4IvAic3GCdk4Gnot+/ChxhHnUjLi4u5sG//51wm6LIYXqJKV/FBgpXTuFnB/elcOUUhX0chNr3pbawP8888wzz58/3upwWbdvvb1Dblh/01SEjKyuL008/naysLKpDLf9KDTMY1KaGL+d94XUpScHLoO8OrKo3XRyd1+g6zrkQsBUobLghM7vYzOaa2dySkpK4FDt//nxqg0ECXfbRIDdx4C9bywnHHcevx/2KE447Dn/ZWq9LSkmBrvvhnOOLL/QG2Bzb2htpvpZ/Nib7vgAAFfJJREFU2L4qZJx44omMGzeOE044gaoUCHqANKPVXUu/I172um/sr6nhf01T1sE59wjwCESuo29+ad930EEH4fen49+yLHLpnMRUqKAr706aBMC7kyYR6nmYtwWlqPQtywAYO3asx5W0bNnZ2QAEwkYjb0ktSo7f8c477+Cc491336Wzv2X/PNsE6uy716m18/LjTjFQf7DuImDNjtYxMz/QBtickOoaaNOmDYcffhiZm5fiq9zoRQkprS6vE5t6Hsb/b+/Ow6Ms7/2Pv7+Tyb6zRRaDEklYGxFIChIQF0RPf7XWWo4VK7XY1h6t/fXnUsVeRasgrj0ercqv1aOtnp7Wii1WWxQVqAFCWLQCCSBg2IQA2RMy233+yNTLo2hZkjzJ5PO6Lq7M5HnmmW+Ya+Yz9/3cz30/v3I7h3LPIZLWz+uSYo4dqSOxupKzxo7VlLknKS2tbXzO4dbu32IsyAqS429k+Z9/R46/kYIYOB0BUNMa99Hr1NN52aJfAww1s9OBPcC/At/4xD5/Aq4GVgJfA95wHk7lN3PmTMrXrsNtXkxr/0IC/c8EdQ21m0haPwIK+PbnHPH7N5G8Zy0pyUl8+5prvK6o2yssLCQxMYHV+xMZkd29r2KYmR97lwcePuJjS62fb14y0etSugTPUip6zv164K/AZuB3zrmNZnaXmX05utuvgN5mtg34EfCpS/A60+DBg3n2mf/kgvPPJ3HvBtIqFuNr9qSDQeSYWGsDKZWvkrRrNcVF43j22WcYOXKk12V1eykpKUyePIXV1UkEwl5XI59Uuj8BB0ybNs3rUroEzXV/glasWMH99z9AfUM9wazBBPsNI5zeXwP1xHvO4WuqJuHAZhJqdpKYmMCNP/gB06dP19z37ai8vJybbrqJmUObmHbqEa/LkaiWENxe1pv+eSN59LHHvC6n02iu+w5QUlLC6NGj+e1vf8vil/9MU+VfcMlZtPYdRrDPGRCX4HWJ0tOEQ/gPbyepugJrOkhiUhIXffn/cMUVV5CTk+N1dTFn7NixFBeN5/dr11DYO0BOSsTrkgT47bZUalqNu77/fa9L6TLUom8Hra2tvPnmm7y4aBFbKiuxuHhae+UR7DOUSGoftfKl4ziHr6WG+INbSTy8DRds5dTcwVz21UuZNm2alqztYAcOHOBbs65mUEIDt42p61br0Mei9w7Hc9+GDGbMmMF1113ndTmdSsvUdqKKigpeeuklXn99KaFQEBLT2tai73Ua4bR+WtFOTp5z+JoO4q/ZSWJdFbTU4fPFUVIyiUsvvZTCwkJ10XeiV199lQULFjD91BauOKNZ3+s98mGzj3vWZ5PRdyC//NVTJCb2rLVHFPQeaGhooLS0lGXLllFWtoZQKIglJNOamUso+7S28/kasS/HykWIazyAv2YnCbVV0NqILy6OMWPGMGXyZCZNmkSvXr28rrJHcs7xyCOPsGjRIr42pJkvn9bidUk9zqEjPu5en00oPp1H/uNRBg8e7HVJnU7n6D2Qnp7OhRdeyIUXXkhzczOrV69m2bJlrFy5itbqSsyfSCBzEKHMQYQyB4I/yeuSpasJB/DX7yWubjeJdbtwgRb8/niKisYzZcoUJkyYQEZGhtdV9nhmxg033EBjYyMvvPYayX7HBYM0OK+z1AeM+9/JooUkfv7Agz0y5P8ZBX0nSElJYerUqUydOpXW1lbKy8tZvnw5b5eW0rj9faDtGvJgxkBCmQOj5/XV2u9xnMPXchh/3W78dXuIazwALkJycgrFE4uZMmUKxcXFOu/eBfl8Pm699Vaampr4dWkpR0LwpcFH1I3fwapbfDz4bhYHAwnc/8AC8vPzvS6pS1LXvYfC4TCVlZWUlZWxctUqtlRW4pzD4pMIpA8glDmIcOZAXLymcYxZoVb89Xvw1+0hoX4vLtAEwJC8PCZ88YsUFRUxcuRI/H59J+8OAoEACxYsYOnSpZwz4AjfzG/Cr+/sHeL9ej8P/z0T50/l7nnzKCws9LokT+kcfTdRW1vL2rVrWb16NatWl1FfVwuAS+1NIH0A4cyBhNNywBfncaVywiIR4pqqiavfQ3z9XnxN1eAcqalpFBWNp6ioiKKiInr3/tTaTdJNRCIRnnrqKX7zm98wuleQ60c1kBwj88d3FWur43l8Uya9+/Rlwf0PkJub63VJnlPQd0ORSIStW7dSVlZG2Zo1bNy4kUg4jMX5CablEMoYSDhjIJHkLF2+15U5h7XWt3XF1+8lofFDXCiAmVFQUMD48eMpLi5m2LBharXHmJdffpmHHnqInOQwN4ysY1CaptA7WREHL+5IZvHOFAqGFTB//r1kZ2d7XVaXoKCPAc3NzWzYsIHy8nLKytawe3fbCr+WkBLt5h9IOGOAuvm7glBr2yC6+r0kNOyFIw0A9O2XwxeLixg3bhxnnXUW6enpHhcqHW39+vXcdedcmhrqmJXfwKT+Aa9L6rbqAsbjGzPYVONn+vTp/PCHPyQpSYOY/0FBH4P2799PeXk55eXlrCkvp7GhLUwiqX0IZgwgnDmIcGo/XcLXGVwEX9Mh/HW7ia/f81F3fHJyCmPHnsX48eMZO3YsAwcO1PXtPdChQ4e46847eefdd5nS/whX5TeRoLNvx6Wixs8vNmfSHI7n//7oR1x00UVel9TlKOhjXCQSYcuWLaxZs4bVZWVs2riRSCSC+RMIpPUnnDmQUOYgXKKWbGwvFmgmrn4P/rrdJDTswwWPYGYMzc/ni8XFjB8/nuHDh6s7XgAIhUI8/fTTPPfccwxMi3DdiHpy1ZX/T4Ui8MedyfzpgxQG9O/PXT+7m7y8PK/L6pIU9D1MY2Mj69atY82aNaxctYqD1dVtG5KzCGQMIJSVSzjtFLX2j4eLENdYTVxtFQkNe7GmQwBkZWVTXNw2gG7s2LFkZWV5XKh0ZatXr+be+fNoqK/j60PaFsPRtLlHt7/ZxxObM3i/Lo5p06Zx4403kpqa6nVZXZaCvgdzzlFVVfXRoL4N69cTDAbbJuyJhn4ocxD4e9Z0kcckHGy79K12Fwl1u3HBFnxxcXxh9OiPRsfn5eWpO16OS21tLffdt4DS0pWM6hXk2uENZCfG1ufwyXAOVuxL5Nfb0ohPTOH/3XQzU6dO9bqsLk9BLx9paWlh7dq1lJaW8re3S9su4TMf4fQcgpm5hLJOxSX13NnWLNCEv3YX/toq4hv24SJhUlJTmThhAmeffTbjx48nLU2nQOTkOOdYvHgxjz36H8QT5OqhDRTnaKBefcB4ujKNtdUJnFlYyO1z5tCvXz+vy+oWFPRyVJFIhM2bN/P222+z4m9/Y1dVFQAupReB7NMJ9h6CS4z9keEWaMZ/eDsJNTvwNbad5sg55RQml5QwceJERo8erXPt0iGqqqqYd8/dVFRuYUJOK9/MbyI1PrY+k4/Vuup4ntqSQUs4jm/PvpbLL7+cuDiNWjxWCno5Jnv27KG0tJQ333yLTZs2Am1T8wZ6DSHU6/TYunQv1Ep8zU7iD28nrn4fAHlnDOXcqedw9tlnM3jwYHXJS6cIhUI8//zzPPPMf5IRH+HaYfWM6hX0uqxO0xKC57amsnxfEnlDTmfOHT9hyJAhXpfV7Sjo5bh9+OGHvPHGGyx57XV27tgOZoTT+xPonUcoezDEJXhd4vELh/DXVuE/vJ34+t0QidB/wECmXXA+5513nmbXEk9VVFQw7567qdq1m2mDWvh6XnPMX4ZXWetnYUUGh1p8XPGNbzBr1izi4+O9LqtbUtDLSdmxYwdLly5lyWuvc2D/h1hcPK298gjmDCeS3PVnpbIj9SQcqCDx8FZcsJXsXr254PzzOO+888jPz1fLXbqM1tZWFi5cyB/+8AcGpDq+N6KO09Jj7zK8UAQW7Ujm5aoUTsnJ4fY5dzB69Givy+rWFPTSLpxzbNq0icWLF/P660sJhYKEM/oT6DucUHZu11pxzzni6veQsH8z/rpd+Hw+SkpKuOSSSygsLNS5P+nSysvLmT/vHmpravjq6U38y+DYuQxvT1McT2zK4IMGHxdffDHXX3+9VmRsBwp6aXe1tbW88sorvLhoUdt1+olptPYpIJAz3Ntu/UiI+OpKkqo3Q0s9mVlZfOWSS/jSl75E3759vatL5DjV19fz0EMP8dZbbzE8O8R3RzTQKzHidVknzDl4a28iz21LJzk1jZtuvoWSkhKvy4oZCnrpMOFwmJUrV/Lii4tYt24tFp9EyymFBPsN69xV9lwE/8FtJO/bAK2NDB8xgq9ddhmTJ0/WOT/ptpxzLFmyhIcfehC/CzC7oJ6z+na/gXpNQePpylTKDiQybuxZ3Hb7HK3Q2M4U9NIpKioqePLJhaxfvw4S02gZMIZQ77yO7dJ3Dn/NByTvXQcttRQMG8b3vvtdxowZ03HPKdLJdu3axZ13zmXbtve5YFALM7rRQL2tdX4e35RJTcDH7NnXMmPGDHyalbPdKeilU5WXl/PkwoVs3bIFktIJ+zvu/JsvfARrqePUU3P5zneuZdKkSRpcJzEpEAiwcOFCXnjhBYZkhLlhVD29k7puV75z8NruJP5rWyr9+vXjJz+dy4gRI7wuK2Yp6KXTOedYvnw5r7zyCsFgx3U1xsXFce6553LBBRdoUhvpEVasWMH8efcQFz7Cv42oY0SvkNclfUprGJ6uSKN0fyITJ07ktttu07LMHUxBLyISQ6qqqvjJHXOo2rWLrw9p4uLcI3SVjqwDLT4eeS+TXY0+rrnm21x55ZXqqu8Enxf0+t8XEelmcnNzefyJJ5k8eQr//X4qv9ycSqgL9OJvrfMzd202NZF07r13AVdddZVCvgvQKyAi0g2lpKQwd+5cZs2axYoPk3jw3QyaQ94169ccSODe9Zlk9unP408+SXFxsWe1yP+moBcR6abMjFmzZvHjH/+YirpE7lmXxeEjnf+x/peqJB59L52hw4bz2C8eZ9CgQZ1eg3w2Bb2ISDc3ffp0Fiy4j0PhFO5en011S+d9tL+4PZnnt6VSMrmEhx/+OVlZWZ323HJsFPQiIjFg3LhxPPzzf+eIL4X5Gzo+7J1rC/mXdqYwffp0fvrTuSQmJnboc8qJUdCLiMSIgoICHnzoYVp9qczrwLB3rm1Rmn+E/C233KL1I7owT4LezHqZ2WtmtjX681NLoJnZmWa20sw2mtm7ZjbDi1pFRLqTgoICHnz4YQK+VO57J4uGQPsP0FuyO4mXdqZw0UUXccstt2hkfRfn1avzY2Cpc24osDR6/5OagW8650YC04Gfm5lO/oiI/BP5+fnMX7CAw8F4/v29DILteOnduup4nt+aSsmkSdx8880K+W7Aq1foEuCZ6O1ngK98cgfn3Bbn3Nbo7b3AAUDLj4mIHINRo0Zx2223s6XWzy83p9Eec6PtqI/j8U0Z5BfkM+eOOxTy3YRXr1KOc24fQPRnv8/b2cyKgATg/U6oTUQkJpx77rnMnj2blfsT+evupJM6VlPQeGRjFpm9+jBv3nySkk7ueNJ5OmxycDN7HTjlKJvmHOdx+gO/Bq52zh21A8rMvgN8B9pmjBIRkTZXXnklmzZu5HerVzIiK0huevi4j+EcPF2ZSm3Ax6N3/UxLzHYznsx1b2aVwDnOuX3RIH/LOVdwlP0ygLeA+c653x/LsTXXvYjI/1ZbW8s135pFUqCGb5zRcNyP39ng54XtKcyePZuZM2d2QIVysj5vrnuvlvv6E3A1cG/05x8/uYOZJQCLgGePNeRFROTTsrKyuH3OHdxy88088E7GCR3jzMJCrrjiinauTDqDVy363sDvgFygCrjcOXfYzMYB33POzTazmcDTwMaPPXSWc27D5x1bLXoRkaPbt28fhw8fPqHH5ufnEx8f384VSXvRMrUiIiIxTMvUioiI9FAKehERkRimoBcREYlhCnoREZEYpqAXERGJYQp6ERGRGKagFxERiWEKehERkRimoBcREYlhCnoREZEYpqAXERGJYQp6ERGRGBZzi9qYWTXwgdd1yAnrAxz0ugiRHkjvve5tsHOu79E2xFzQS/dmZuWftQKTiHQcvfdil7ruRUREYpiCXkREJIYp6KWrWeh1ASI9lN57MUrn6EVERGKYWvQiIiIxTEEvIiISwxT0IiIiMUxBL53KzH5mZjd+7P49ZvYDM7vZzNaY2btmdmd0W6qZ/dnM3jGz98xshneVi8QOMzvNzDab2f83s41mtsTMks3sTDNbFX0fLjKzbK9rlZOnoJfO9ivgagAz8wH/CuwHhgJFwJnAWDObDEwH9jrnCp1zo4C/eFOySEwaCjzmnBsJ1AKXAc8CtzrnvgD8Hfiph/VJO1HQS6dyzu0EDpnZGGAasB4Y/7Hb64BhtH0I/R0438wWmFmJc67Om6pFYtIO59yG6O21QB6Q5ZxbFv3dM8BkTyqTduX3ugDpkX4JzAJOAZ4CzgPmO+ee/OSOZjYWuBiYb2ZLnHN3dWahIjGs9WO3w0CWV4VIx1KLXrywiLZu+fHAX6P/rjGzNAAzG2hm/cxsANDsnPsN8ABwllcFi/QAdUCNmZVE718FLPuc/aWbUIteOp1zLmBmbwK1zrkwsMTMhgMrzQygEZgJnAHcb2YRIAhc51XNIj3E1cATZpYCbAe+5XE90g40M550uuggvHXA5c65rV7XIyISy9R1L53KzEYA24ClCnkRkY6nFr2IiEgMU4teREQkhinoRUREYpiCXkREJIYp6EV6IDPLMrPvd8LznGNmE0/gcTvNrE9H1CTS0yjoRXqmLOCYg97anMjnxTnAcQe9iLQfTZgj0jPdC+SZ2QbgTeALQDYQD9zhnPujmZ0GvBrdPgH4ipmdD9wK7AW2Aq3OuevNrC/wBJAbPf4PgT3A94Cwmc0EbgAqPrmfc+5tM+sN/BfQFygDrAP/dpEeRZfXifRA0RB/2Tk3ysz8QIpzrj7aXb6KtkWFBtM2O9pE59yq6JTEpbRNRdwAvAG8Ew3654FfOOf+Zma5wF+dc8PNbC7Q6Jx7IPq8n7XfI8BB59xdZvYvwMtAX+fcwU77TxGJUWrRi4gB86JLA0eAgUBOdNsHzrlV0dtFwDLn3GEAM/s9kB/ddj4wIjqFMUCGmaUf5bk+a7/JwFcBnHN/NrOa9vrjRHo6Bb2IXElbl/lY51zQzHYCSdFtTR/b7/O6033ABOdcy8d/+bFAP5b91L0o0gE0GE+kZ2oA/tHizgQOREN+Km1d9kdTBkwxs+xod/9lH9u2BLj+H3fM7MyjPM/n7becti8cmNlFtI0XEJF2oKAX6YGcc4eAt83sPeBMYJyZldMWthWf8Zg9wDxgNfA6sIm2pU0BfhA9xrtmtom2QXgAi4FLzWxDdPnTz9rvTmCyma0DpgFV7fsXi/RcGownIsfMzNKcc43RFv0i4Cnn3CKv6xKRz6YWvYgcj7nRS/LeA3YAL3lcj4j8E2rRi4iIxDC16EVERGKYgl5ERCSGKehFRERimIJeREQkhinoRUREYpiCXkREJIb9DwSGLosqObtfAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 576x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 576x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 576x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 576x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 576x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 576x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 576x432 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "from numpy import median\n",
    "for col in obj_col[1:]:\n",
    "    plt.figure(figsize=(8,6))\n",
    "    sns.violinplot(bm1[col],bm1[\"response\"])\n",
    "    plt.title(\"Response vs \"+col,fontsize=15)\n",
    "    plt.xlabel(col,fontsize=10)\n",
    "    plt.ylabel(\"Response\",fontsize=10)\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 576x432 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(8,6))\n",
    "sns.heatmap(bm1.corr(),annot=True,cmap='RdBu_r')\n",
    "plt.title(\"Correlation Of Each Numerical Features\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>job</th>\n",
       "      <th>marital</th>\n",
       "      <th>education</th>\n",
       "      <th>targeted</th>\n",
       "      <th>default</th>\n",
       "      <th>housing</th>\n",
       "      <th>loan</th>\n",
       "      <th>contact</th>\n",
       "      <th>month</th>\n",
       "      <th>poutcome</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>24060</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24062</th>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24064</th>\n",
       "      <td>7</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24072</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24077</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       job  marital  education  targeted  default  housing  loan  contact  \\\n",
       "24060    0        1          2         1        0        0     0        1   \n",
       "24062    0        2          1         1        0        1     1        1   \n",
       "24064    7        1          1         1        0        1     0        1   \n",
       "24072    4        1          2         1        0        1     0        1   \n",
       "24077    4        1          2         1        0        1     0        1   \n",
       "\n",
       "       month  poutcome  \n",
       "24060     10         0  \n",
       "24062     10         1  \n",
       "24064     10         0  \n",
       "24072     10         1  \n",
       "24077     10         0  "
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.preprocessing import LabelEncoder\n",
    "bm2 = bm1[obj_col].apply(LabelEncoder().fit_transform)\n",
    "bm2.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>job</th>\n",
       "      <th>marital</th>\n",
       "      <th>education</th>\n",
       "      <th>targeted</th>\n",
       "      <th>default</th>\n",
       "      <th>housing</th>\n",
       "      <th>loan</th>\n",
       "      <th>contact</th>\n",
       "      <th>month</th>\n",
       "      <th>poutcome</th>\n",
       "      <th>age</th>\n",
       "      <th>salary</th>\n",
       "      <th>balance</th>\n",
       "      <th>day</th>\n",
       "      <th>duration</th>\n",
       "      <th>campaign</th>\n",
       "      <th>pdays</th>\n",
       "      <th>previous</th>\n",
       "      <th>response</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>24060</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>0</td>\n",
       "      <td>33</td>\n",
       "      <td>50000</td>\n",
       "      <td>882</td>\n",
       "      <td>21</td>\n",
       "      <td>39</td>\n",
       "      <td>1</td>\n",
       "      <td>151</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24062</th>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>1</td>\n",
       "      <td>42</td>\n",
       "      <td>50000</td>\n",
       "      <td>-247</td>\n",
       "      <td>21</td>\n",
       "      <td>519</td>\n",
       "      <td>1</td>\n",
       "      <td>166</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24064</th>\n",
       "      <td>7</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>0</td>\n",
       "      <td>33</td>\n",
       "      <td>70000</td>\n",
       "      <td>3444</td>\n",
       "      <td>21</td>\n",
       "      <td>144</td>\n",
       "      <td>1</td>\n",
       "      <td>91</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24072</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>1</td>\n",
       "      <td>36</td>\n",
       "      <td>100000</td>\n",
       "      <td>2415</td>\n",
       "      <td>22</td>\n",
       "      <td>73</td>\n",
       "      <td>1</td>\n",
       "      <td>86</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24077</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>0</td>\n",
       "      <td>36</td>\n",
       "      <td>100000</td>\n",
       "      <td>0</td>\n",
       "      <td>23</td>\n",
       "      <td>140</td>\n",
       "      <td>1</td>\n",
       "      <td>143</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       job  marital  education  targeted  default  housing  loan  contact  \\\n",
       "24060    0        1          2         1        0        0     0        1   \n",
       "24062    0        2          1         1        0        1     1        1   \n",
       "24064    7        1          1         1        0        1     0        1   \n",
       "24072    4        1          2         1        0        1     0        1   \n",
       "24077    4        1          2         1        0        1     0        1   \n",
       "\n",
       "       month  poutcome  age  salary  balance  day  duration  campaign  pdays  \\\n",
       "24060     10         0   33   50000      882   21        39         1    151   \n",
       "24062     10         1   42   50000     -247   21       519         1    166   \n",
       "24064     10         0   33   70000     3444   21       144         1     91   \n",
       "24072     10         1   36  100000     2415   22        73         1     86   \n",
       "24077     10         0   36  100000        0   23       140         1    143   \n",
       "\n",
       "       previous  response  \n",
       "24060         3         0  \n",
       "24062         1         1  \n",
       "24064         4         1  \n",
       "24072         4         0  \n",
       "24077         3         1  "
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "bm3 = bm2.join(bm1[num_col])\n",
    "bm3.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>job</th>\n",
       "      <th>marital</th>\n",
       "      <th>education</th>\n",
       "      <th>targeted</th>\n",
       "      <th>default</th>\n",
       "      <th>housing</th>\n",
       "      <th>loan</th>\n",
       "      <th>contact</th>\n",
       "      <th>month</th>\n",
       "      <th>poutcome</th>\n",
       "      <th>age</th>\n",
       "      <th>salary</th>\n",
       "      <th>balance</th>\n",
       "      <th>day</th>\n",
       "      <th>duration</th>\n",
       "      <th>campaign</th>\n",
       "      <th>pdays</th>\n",
       "      <th>previous</th>\n",
       "      <th>response</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>job</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.072746</td>\n",
       "      <td>0.159280</td>\n",
       "      <td>-0.091197</td>\n",
       "      <td>-0.021012</td>\n",
       "      <td>-0.132378</td>\n",
       "      <td>-0.033500</td>\n",
       "      <td>-0.006279</td>\n",
       "      <td>-0.002498</td>\n",
       "      <td>0.066642</td>\n",
       "      <td>-0.020606</td>\n",
       "      <td>0.115271</td>\n",
       "      <td>0.041975</td>\n",
       "      <td>0.013841</td>\n",
       "      <td>0.024449</td>\n",
       "      <td>-0.008764</td>\n",
       "      <td>-0.110505</td>\n",
       "      <td>-0.000266</td>\n",
       "      <td>0.081239</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>marital</th>\n",
       "      <td>0.072746</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.121688</td>\n",
       "      <td>-0.255247</td>\n",
       "      <td>-0.005632</td>\n",
       "      <td>-0.056963</td>\n",
       "      <td>-0.055435</td>\n",
       "      <td>-0.031866</td>\n",
       "      <td>-0.024130</td>\n",
       "      <td>0.045015</td>\n",
       "      <td>-0.414972</td>\n",
       "      <td>-0.042212</td>\n",
       "      <td>-0.019767</td>\n",
       "      <td>0.016797</td>\n",
       "      <td>-0.009075</td>\n",
       "      <td>-0.008338</td>\n",
       "      <td>-0.011861</td>\n",
       "      <td>0.004536</td>\n",
       "      <td>0.049234</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>education</th>\n",
       "      <td>0.159280</td>\n",
       "      <td>0.121688</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.632513</td>\n",
       "      <td>-0.013682</td>\n",
       "      <td>-0.129804</td>\n",
       "      <td>-0.047718</td>\n",
       "      <td>-0.048456</td>\n",
       "      <td>-0.010829</td>\n",
       "      <td>0.082852</td>\n",
       "      <td>-0.119516</td>\n",
       "      <td>0.423157</td>\n",
       "      <td>0.074166</td>\n",
       "      <td>0.023542</td>\n",
       "      <td>-0.001142</td>\n",
       "      <td>-0.024343</td>\n",
       "      <td>-0.140155</td>\n",
       "      <td>0.000115</td>\n",
       "      <td>0.108098</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>targeted</th>\n",
       "      <td>-0.091197</td>\n",
       "      <td>-0.255247</td>\n",
       "      <td>-0.632513</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.019593</td>\n",
       "      <td>0.087470</td>\n",
       "      <td>0.070420</td>\n",
       "      <td>0.020464</td>\n",
       "      <td>0.010187</td>\n",
       "      <td>-0.071356</td>\n",
       "      <td>0.140750</td>\n",
       "      <td>-0.228338</td>\n",
       "      <td>-0.052007</td>\n",
       "      <td>-0.026179</td>\n",
       "      <td>-0.014729</td>\n",
       "      <td>0.017948</td>\n",
       "      <td>0.075638</td>\n",
       "      <td>-0.001205</td>\n",
       "      <td>-0.091216</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>default</th>\n",
       "      <td>-0.021012</td>\n",
       "      <td>-0.005632</td>\n",
       "      <td>-0.013682</td>\n",
       "      <td>0.019593</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.022644</td>\n",
       "      <td>0.052261</td>\n",
       "      <td>-0.019834</td>\n",
       "      <td>0.020123</td>\n",
       "      <td>-0.025566</td>\n",
       "      <td>-0.027825</td>\n",
       "      <td>0.000361</td>\n",
       "      <td>-0.045010</td>\n",
       "      <td>-0.001013</td>\n",
       "      <td>-0.002635</td>\n",
       "      <td>-0.002064</td>\n",
       "      <td>0.033760</td>\n",
       "      <td>0.012149</td>\n",
       "      <td>-0.028299</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>housing</th>\n",
       "      <td>-0.132378</td>\n",
       "      <td>-0.056963</td>\n",
       "      <td>-0.129804</td>\n",
       "      <td>0.087470</td>\n",
       "      <td>0.022644</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.109815</td>\n",
       "      <td>-0.069869</td>\n",
       "      <td>0.014452</td>\n",
       "      <td>-0.284439</td>\n",
       "      <td>-0.179386</td>\n",
       "      <td>-0.035905</td>\n",
       "      <td>-0.109163</td>\n",
       "      <td>-0.066740</td>\n",
       "      <td>-0.072070</td>\n",
       "      <td>0.063071</td>\n",
       "      <td>0.335124</td>\n",
       "      <td>0.008934</td>\n",
       "      <td>-0.317501</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>loan</th>\n",
       "      <td>-0.033500</td>\n",
       "      <td>-0.055435</td>\n",
       "      <td>-0.047718</td>\n",
       "      <td>0.070420</td>\n",
       "      <td>0.052261</td>\n",
       "      <td>0.109815</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.020904</td>\n",
       "      <td>0.000524</td>\n",
       "      <td>-0.103154</td>\n",
       "      <td>-0.008330</td>\n",
       "      <td>0.013788</td>\n",
       "      <td>-0.085004</td>\n",
       "      <td>0.007550</td>\n",
       "      <td>-0.033874</td>\n",
       "      <td>0.007444</td>\n",
       "      <td>0.022454</td>\n",
       "      <td>0.016549</td>\n",
       "      <td>-0.115805</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>contact</th>\n",
       "      <td>-0.006279</td>\n",
       "      <td>-0.031866</td>\n",
       "      <td>-0.048456</td>\n",
       "      <td>0.020464</td>\n",
       "      <td>-0.019834</td>\n",
       "      <td>-0.069869</td>\n",
       "      <td>-0.020904</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.047045</td>\n",
       "      <td>0.024590</td>\n",
       "      <td>0.166384</td>\n",
       "      <td>-0.035805</td>\n",
       "      <td>0.030317</td>\n",
       "      <td>-0.012330</td>\n",
       "      <td>-0.036360</td>\n",
       "      <td>0.063199</td>\n",
       "      <td>0.077235</td>\n",
       "      <td>0.043830</td>\n",
       "      <td>-0.014321</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>month</th>\n",
       "      <td>-0.002498</td>\n",
       "      <td>-0.024130</td>\n",
       "      <td>-0.010829</td>\n",
       "      <td>0.010187</td>\n",
       "      <td>0.020123</td>\n",
       "      <td>0.014452</td>\n",
       "      <td>0.000524</td>\n",
       "      <td>0.047045</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.019188</td>\n",
       "      <td>0.013909</td>\n",
       "      <td>0.019820</td>\n",
       "      <td>0.015723</td>\n",
       "      <td>-0.024264</td>\n",
       "      <td>-0.036446</td>\n",
       "      <td>0.023224</td>\n",
       "      <td>0.022529</td>\n",
       "      <td>0.013977</td>\n",
       "      <td>-0.009586</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>poutcome</th>\n",
       "      <td>0.066642</td>\n",
       "      <td>0.045015</td>\n",
       "      <td>0.082852</td>\n",
       "      <td>-0.071356</td>\n",
       "      <td>-0.025566</td>\n",
       "      <td>-0.284439</td>\n",
       "      <td>-0.103154</td>\n",
       "      <td>0.024590</td>\n",
       "      <td>-0.019188</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.051987</td>\n",
       "      <td>0.013079</td>\n",
       "      <td>0.051464</td>\n",
       "      <td>0.017125</td>\n",
       "      <td>0.108473</td>\n",
       "      <td>-0.000502</td>\n",
       "      <td>-0.235119</td>\n",
       "      <td>0.041607</td>\n",
       "      <td>0.420519</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>age</th>\n",
       "      <td>-0.020606</td>\n",
       "      <td>-0.414972</td>\n",
       "      <td>-0.119516</td>\n",
       "      <td>0.140750</td>\n",
       "      <td>-0.027825</td>\n",
       "      <td>-0.179386</td>\n",
       "      <td>-0.008330</td>\n",
       "      <td>0.166384</td>\n",
       "      <td>0.013909</td>\n",
       "      <td>0.051987</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.058322</td>\n",
       "      <td>0.115480</td>\n",
       "      <td>0.018399</td>\n",
       "      <td>0.060519</td>\n",
       "      <td>0.003842</td>\n",
       "      <td>-0.107863</td>\n",
       "      <td>0.002280</td>\n",
       "      <td>0.072577</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>salary</th>\n",
       "      <td>0.115271</td>\n",
       "      <td>-0.042212</td>\n",
       "      <td>0.423157</td>\n",
       "      <td>-0.228338</td>\n",
       "      <td>0.000361</td>\n",
       "      <td>-0.035905</td>\n",
       "      <td>0.013788</td>\n",
       "      <td>-0.035805</td>\n",
       "      <td>0.019820</td>\n",
       "      <td>0.013079</td>\n",
       "      <td>0.058322</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.066079</td>\n",
       "      <td>0.002379</td>\n",
       "      <td>0.005734</td>\n",
       "      <td>-0.018559</td>\n",
       "      <td>-0.106134</td>\n",
       "      <td>0.028427</td>\n",
       "      <td>0.036774</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>balance</th>\n",
       "      <td>0.041975</td>\n",
       "      <td>-0.019767</td>\n",
       "      <td>0.074166</td>\n",
       "      <td>-0.052007</td>\n",
       "      <td>-0.045010</td>\n",
       "      <td>-0.109163</td>\n",
       "      <td>-0.085004</td>\n",
       "      <td>0.030317</td>\n",
       "      <td>0.015723</td>\n",
       "      <td>0.051464</td>\n",
       "      <td>0.115480</td>\n",
       "      <td>0.066079</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.051732</td>\n",
       "      <td>0.040998</td>\n",
       "      <td>-0.008150</td>\n",
       "      <td>-0.108122</td>\n",
       "      <td>0.001570</td>\n",
       "      <td>0.076995</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>day</th>\n",
       "      <td>0.013841</td>\n",
       "      <td>0.016797</td>\n",
       "      <td>0.023542</td>\n",
       "      <td>-0.026179</td>\n",
       "      <td>-0.001013</td>\n",
       "      <td>-0.066740</td>\n",
       "      <td>0.007550</td>\n",
       "      <td>-0.012330</td>\n",
       "      <td>-0.024264</td>\n",
       "      <td>0.017125</td>\n",
       "      <td>0.018399</td>\n",
       "      <td>0.002379</td>\n",
       "      <td>0.051732</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.012687</td>\n",
       "      <td>-0.029295</td>\n",
       "      <td>-0.090095</td>\n",
       "      <td>-0.016801</td>\n",
       "      <td>0.029952</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>duration</th>\n",
       "      <td>0.024449</td>\n",
       "      <td>-0.009075</td>\n",
       "      <td>-0.001142</td>\n",
       "      <td>-0.014729</td>\n",
       "      <td>-0.002635</td>\n",
       "      <td>-0.072070</td>\n",
       "      <td>-0.033874</td>\n",
       "      <td>-0.036360</td>\n",
       "      <td>-0.036446</td>\n",
       "      <td>0.108473</td>\n",
       "      <td>0.060519</td>\n",
       "      <td>0.005734</td>\n",
       "      <td>0.040998</td>\n",
       "      <td>-0.012687</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.088394</td>\n",
       "      <td>-0.024407</td>\n",
       "      <td>-0.002150</td>\n",
       "      <td>0.342610</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>campaign</th>\n",
       "      <td>-0.008764</td>\n",
       "      <td>-0.008338</td>\n",
       "      <td>-0.024343</td>\n",
       "      <td>0.017948</td>\n",
       "      <td>-0.002064</td>\n",
       "      <td>0.063071</td>\n",
       "      <td>0.007444</td>\n",
       "      <td>0.063199</td>\n",
       "      <td>0.023224</td>\n",
       "      <td>-0.000502</td>\n",
       "      <td>0.003842</td>\n",
       "      <td>-0.018559</td>\n",
       "      <td>-0.008150</td>\n",
       "      <td>-0.029295</td>\n",
       "      <td>-0.088394</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.050534</td>\n",
       "      <td>0.135523</td>\n",
       "      <td>-0.094110</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>pdays</th>\n",
       "      <td>-0.110505</td>\n",
       "      <td>-0.011861</td>\n",
       "      <td>-0.140155</td>\n",
       "      <td>0.075638</td>\n",
       "      <td>0.033760</td>\n",
       "      <td>0.335124</td>\n",
       "      <td>0.022454</td>\n",
       "      <td>0.077235</td>\n",
       "      <td>0.022529</td>\n",
       "      <td>-0.235119</td>\n",
       "      <td>-0.107863</td>\n",
       "      <td>-0.106134</td>\n",
       "      <td>-0.108122</td>\n",
       "      <td>-0.090095</td>\n",
       "      <td>-0.024407</td>\n",
       "      <td>0.050534</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.021885</td>\n",
       "      <td>-0.152206</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>previous</th>\n",
       "      <td>-0.000266</td>\n",
       "      <td>0.004536</td>\n",
       "      <td>0.000115</td>\n",
       "      <td>-0.001205</td>\n",
       "      <td>0.012149</td>\n",
       "      <td>0.008934</td>\n",
       "      <td>0.016549</td>\n",
       "      <td>0.043830</td>\n",
       "      <td>0.013977</td>\n",
       "      <td>0.041607</td>\n",
       "      <td>0.002280</td>\n",
       "      <td>0.028427</td>\n",
       "      <td>0.001570</td>\n",
       "      <td>-0.016801</td>\n",
       "      <td>-0.002150</td>\n",
       "      <td>0.135523</td>\n",
       "      <td>-0.021885</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.008622</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>response</th>\n",
       "      <td>0.081239</td>\n",
       "      <td>0.049234</td>\n",
       "      <td>0.108098</td>\n",
       "      <td>-0.091216</td>\n",
       "      <td>-0.028299</td>\n",
       "      <td>-0.317501</td>\n",
       "      <td>-0.115805</td>\n",
       "      <td>-0.014321</td>\n",
       "      <td>-0.009586</td>\n",
       "      <td>0.420519</td>\n",
       "      <td>0.072577</td>\n",
       "      <td>0.036774</td>\n",
       "      <td>0.076995</td>\n",
       "      <td>0.029952</td>\n",
       "      <td>0.342610</td>\n",
       "      <td>-0.094110</td>\n",
       "      <td>-0.152206</td>\n",
       "      <td>0.008622</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                job   marital  education  targeted   default   housing  \\\n",
       "job        1.000000  0.072746   0.159280 -0.091197 -0.021012 -0.132378   \n",
       "marital    0.072746  1.000000   0.121688 -0.255247 -0.005632 -0.056963   \n",
       "education  0.159280  0.121688   1.000000 -0.632513 -0.013682 -0.129804   \n",
       "targeted  -0.091197 -0.255247  -0.632513  1.000000  0.019593  0.087470   \n",
       "default   -0.021012 -0.005632  -0.013682  0.019593  1.000000  0.022644   \n",
       "housing   -0.132378 -0.056963  -0.129804  0.087470  0.022644  1.000000   \n",
       "loan      -0.033500 -0.055435  -0.047718  0.070420  0.052261  0.109815   \n",
       "contact   -0.006279 -0.031866  -0.048456  0.020464 -0.019834 -0.069869   \n",
       "month     -0.002498 -0.024130  -0.010829  0.010187  0.020123  0.014452   \n",
       "poutcome   0.066642  0.045015   0.082852 -0.071356 -0.025566 -0.284439   \n",
       "age       -0.020606 -0.414972  -0.119516  0.140750 -0.027825 -0.179386   \n",
       "salary     0.115271 -0.042212   0.423157 -0.228338  0.000361 -0.035905   \n",
       "balance    0.041975 -0.019767   0.074166 -0.052007 -0.045010 -0.109163   \n",
       "day        0.013841  0.016797   0.023542 -0.026179 -0.001013 -0.066740   \n",
       "duration   0.024449 -0.009075  -0.001142 -0.014729 -0.002635 -0.072070   \n",
       "campaign  -0.008764 -0.008338  -0.024343  0.017948 -0.002064  0.063071   \n",
       "pdays     -0.110505 -0.011861  -0.140155  0.075638  0.033760  0.335124   \n",
       "previous  -0.000266  0.004536   0.000115 -0.001205  0.012149  0.008934   \n",
       "response   0.081239  0.049234   0.108098 -0.091216 -0.028299 -0.317501   \n",
       "\n",
       "               loan   contact     month  poutcome       age    salary  \\\n",
       "job       -0.033500 -0.006279 -0.002498  0.066642 -0.020606  0.115271   \n",
       "marital   -0.055435 -0.031866 -0.024130  0.045015 -0.414972 -0.042212   \n",
       "education -0.047718 -0.048456 -0.010829  0.082852 -0.119516  0.423157   \n",
       "targeted   0.070420  0.020464  0.010187 -0.071356  0.140750 -0.228338   \n",
       "default    0.052261 -0.019834  0.020123 -0.025566 -0.027825  0.000361   \n",
       "housing    0.109815 -0.069869  0.014452 -0.284439 -0.179386 -0.035905   \n",
       "loan       1.000000 -0.020904  0.000524 -0.103154 -0.008330  0.013788   \n",
       "contact   -0.020904  1.000000  0.047045  0.024590  0.166384 -0.035805   \n",
       "month      0.000524  0.047045  1.000000 -0.019188  0.013909  0.019820   \n",
       "poutcome  -0.103154  0.024590 -0.019188  1.000000  0.051987  0.013079   \n",
       "age       -0.008330  0.166384  0.013909  0.051987  1.000000  0.058322   \n",
       "salary     0.013788 -0.035805  0.019820  0.013079  0.058322  1.000000   \n",
       "balance   -0.085004  0.030317  0.015723  0.051464  0.115480  0.066079   \n",
       "day        0.007550 -0.012330 -0.024264  0.017125  0.018399  0.002379   \n",
       "duration  -0.033874 -0.036360 -0.036446  0.108473  0.060519  0.005734   \n",
       "campaign   0.007444  0.063199  0.023224 -0.000502  0.003842 -0.018559   \n",
       "pdays      0.022454  0.077235  0.022529 -0.235119 -0.107863 -0.106134   \n",
       "previous   0.016549  0.043830  0.013977  0.041607  0.002280  0.028427   \n",
       "response  -0.115805 -0.014321 -0.009586  0.420519  0.072577  0.036774   \n",
       "\n",
       "            balance       day  duration  campaign     pdays  previous  \\\n",
       "job        0.041975  0.013841  0.024449 -0.008764 -0.110505 -0.000266   \n",
       "marital   -0.019767  0.016797 -0.009075 -0.008338 -0.011861  0.004536   \n",
       "education  0.074166  0.023542 -0.001142 -0.024343 -0.140155  0.000115   \n",
       "targeted  -0.052007 -0.026179 -0.014729  0.017948  0.075638 -0.001205   \n",
       "default   -0.045010 -0.001013 -0.002635 -0.002064  0.033760  0.012149   \n",
       "housing   -0.109163 -0.066740 -0.072070  0.063071  0.335124  0.008934   \n",
       "loan      -0.085004  0.007550 -0.033874  0.007444  0.022454  0.016549   \n",
       "contact    0.030317 -0.012330 -0.036360  0.063199  0.077235  0.043830   \n",
       "month      0.015723 -0.024264 -0.036446  0.023224  0.022529  0.013977   \n",
       "poutcome   0.051464  0.017125  0.108473 -0.000502 -0.235119  0.041607   \n",
       "age        0.115480  0.018399  0.060519  0.003842 -0.107863  0.002280   \n",
       "salary     0.066079  0.002379  0.005734 -0.018559 -0.106134  0.028427   \n",
       "balance    1.000000  0.051732  0.040998 -0.008150 -0.108122  0.001570   \n",
       "day        0.051732  1.000000 -0.012687 -0.029295 -0.090095 -0.016801   \n",
       "duration   0.040998 -0.012687  1.000000 -0.088394 -0.024407 -0.002150   \n",
       "campaign  -0.008150 -0.029295 -0.088394  1.000000  0.050534  0.135523   \n",
       "pdays     -0.108122 -0.090095 -0.024407  0.050534  1.000000 -0.021885   \n",
       "previous   0.001570 -0.016801 -0.002150  0.135523 -0.021885  1.000000   \n",
       "response   0.076995  0.029952  0.342610 -0.094110 -0.152206  0.008622   \n",
       "\n",
       "           response  \n",
       "job        0.081239  \n",
       "marital    0.049234  \n",
       "education  0.108098  \n",
       "targeted  -0.091216  \n",
       "default   -0.028299  \n",
       "housing   -0.317501  \n",
       "loan      -0.115805  \n",
       "contact   -0.014321  \n",
       "month     -0.009586  \n",
       "poutcome   0.420519  \n",
       "age        0.072577  \n",
       "salary     0.036774  \n",
       "balance    0.076995  \n",
       "day        0.029952  \n",
       "duration   0.342610  \n",
       "campaign  -0.094110  \n",
       "pdays     -0.152206  \n",
       "previous   0.008622  \n",
       "response   1.000000  "
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "bm3.corr()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>job</th>\n",
       "      <th>marital</th>\n",
       "      <th>education</th>\n",
       "      <th>targeted</th>\n",
       "      <th>default</th>\n",
       "      <th>housing</th>\n",
       "      <th>loan</th>\n",
       "      <th>contact</th>\n",
       "      <th>month</th>\n",
       "      <th>poutcome</th>\n",
       "      <th>age</th>\n",
       "      <th>salary</th>\n",
       "      <th>balance</th>\n",
       "      <th>day</th>\n",
       "      <th>duration</th>\n",
       "      <th>campaign</th>\n",
       "      <th>pdays</th>\n",
       "      <th>previous</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>24060</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>0</td>\n",
       "      <td>33</td>\n",
       "      <td>50000</td>\n",
       "      <td>882</td>\n",
       "      <td>21</td>\n",
       "      <td>39</td>\n",
       "      <td>1</td>\n",
       "      <td>151</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24062</th>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>1</td>\n",
       "      <td>42</td>\n",
       "      <td>50000</td>\n",
       "      <td>-247</td>\n",
       "      <td>21</td>\n",
       "      <td>519</td>\n",
       "      <td>1</td>\n",
       "      <td>166</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24064</th>\n",
       "      <td>7</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>0</td>\n",
       "      <td>33</td>\n",
       "      <td>70000</td>\n",
       "      <td>3444</td>\n",
       "      <td>21</td>\n",
       "      <td>144</td>\n",
       "      <td>1</td>\n",
       "      <td>91</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24072</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>1</td>\n",
       "      <td>36</td>\n",
       "      <td>100000</td>\n",
       "      <td>2415</td>\n",
       "      <td>22</td>\n",
       "      <td>73</td>\n",
       "      <td>1</td>\n",
       "      <td>86</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24077</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>10</td>\n",
       "      <td>0</td>\n",
       "      <td>36</td>\n",
       "      <td>100000</td>\n",
       "      <td>0</td>\n",
       "      <td>23</td>\n",
       "      <td>140</td>\n",
       "      <td>1</td>\n",
       "      <td>143</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       job  marital  education  targeted  default  housing  loan  contact  \\\n",
       "24060    0        1          2         1        0        0     0        1   \n",
       "24062    0        2          1         1        0        1     1        1   \n",
       "24064    7        1          1         1        0        1     0        1   \n",
       "24072    4        1          2         1        0        1     0        1   \n",
       "24077    4        1          2         1        0        1     0        1   \n",
       "\n",
       "       month  poutcome  age  salary  balance  day  duration  campaign  pdays  \\\n",
       "24060     10         0   33   50000      882   21        39         1    151   \n",
       "24062     10         1   42   50000     -247   21       519         1    166   \n",
       "24064     10         0   33   70000     3444   21       144         1     91   \n",
       "24072     10         1   36  100000     2415   22        73         1     86   \n",
       "24077     10         0   36  100000        0   23       140         1    143   \n",
       "\n",
       "       previous  \n",
       "24060         3  \n",
       "24062         1  \n",
       "24064         4  \n",
       "24072         4  \n",
       "24077         3  "
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.model_selection import cross_val_score\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import classification_report\n",
    "from sklearn.metrics import confusion_matrix\n",
    "from sklearn.metrics import f1_score\n",
    "np.random.seed(42)\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")\n",
    "X = bm3.drop(\"response\", axis=1)\n",
    "X.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>response</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>24060</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24062</th>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24064</th>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24072</th>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24077</th>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       response\n",
       "24060         0\n",
       "24062         1\n",
       "24064         1\n",
       "24072         0\n",
       "24077         1"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y= bm3[['response']]\n",
    "y.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,\n",
       "                   intercept_scaling=1, l1_ratio=None, max_iter=100,\n",
       "                   multi_class='auto', n_jobs=None, penalty='l2',\n",
       "                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,\n",
       "                   warm_start=False)"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=42)\n",
    "lr = LogisticRegression()\n",
    "lr.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7781983345950038"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cv_score= cross_val_score(lr,X_train,y_train, cv=5)\n",
    "np.mean(cv_score)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.81      0.96      0.87      1279\n",
      "           1       0.58      0.21      0.30       373\n",
      "\n",
      "    accuracy                           0.79      1652\n",
      "   macro avg       0.69      0.58      0.59      1652\n",
      "weighted avg       0.76      0.79      0.75      1652\n",
      "\n"
     ]
    }
   ],
   "source": [
    "y_pred = lr.predict(X_test)\n",
    "print(classification_report(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1224,  296],\n",
       "       [  55,   77]], dtype=int64)"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "confusion_matrix(y_pred,y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.30495049504950494"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "f1_score(y_pred,y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RFE(estimator=LogisticRegression(C=1.0, class_weight=None, dual=False,\n",
       "                                 fit_intercept=True, intercept_scaling=1,\n",
       "                                 l1_ratio=None, max_iter=100,\n",
       "                                 multi_class='auto', n_jobs=None, penalty='l2',\n",
       "                                 random_state=None, solver='lbfgs', tol=0.0001,\n",
       "                                 verbose=0, warm_start=False),\n",
       "    n_features_to_select=5, step=1, verbose=0)"
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.feature_selection import RFE\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "scaler = MinMaxScaler()\n",
    "rfe = RFE(lr, 5)\n",
    "rfe.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([False, False, False, False,  True,  True,  True,  True, False,\n",
       "        True, False, False, False, False, False, False, False, False])"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rfe.support_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['default', 'housing', 'loan', 'contact', 'poutcome'], dtype='object')"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train.columns[rfe.support_]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,\n",
       "                   intercept_scaling=1, l1_ratio=None, max_iter=100,\n",
       "                   multi_class='auto', n_jobs=None, penalty='l2',\n",
       "                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,\n",
       "                   warm_start=False)"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cols = X_train.columns[rfe.support_]\n",
    "lr.fit(X_train[cols],y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.5043478260869565"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred2 = lr.predict(X_test[cols])\n",
    "f1_score(y_pred2,y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1222,  228],\n",
       "       [  57,  145]], dtype=int64)"
      ]
     },
     "execution_count": 38,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "confusion_matrix(y_pred2,y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>job</th>\n",
       "      <th>marital</th>\n",
       "      <th>education</th>\n",
       "      <th>targeted</th>\n",
       "      <th>default</th>\n",
       "      <th>housing</th>\n",
       "      <th>loan</th>\n",
       "      <th>contact</th>\n",
       "      <th>month</th>\n",
       "      <th>poutcome</th>\n",
       "      <th>age</th>\n",
       "      <th>salary</th>\n",
       "      <th>balance</th>\n",
       "      <th>day</th>\n",
       "      <th>duration</th>\n",
       "      <th>campaign</th>\n",
       "      <th>pdays</th>\n",
       "      <th>previous</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>40829</th>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>40</td>\n",
       "      <td>50000</td>\n",
       "      <td>100</td>\n",
       "      <td>11</td>\n",
       "      <td>221</td>\n",
       "      <td>1</td>\n",
       "      <td>461</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>29715</th>\n",
       "      <td>9</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>45</td>\n",
       "      <td>60000</td>\n",
       "      <td>366</td>\n",
       "      <td>3</td>\n",
       "      <td>235</td>\n",
       "      <td>2</td>\n",
       "      <td>169</td>\n",
       "      <td>12</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>35015</th>\n",
       "      <td>9</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>0</td>\n",
       "      <td>32</td>\n",
       "      <td>60000</td>\n",
       "      <td>-360</td>\n",
       "      <td>6</td>\n",
       "      <td>131</td>\n",
       "      <td>2</td>\n",
       "      <td>344</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>41008</th>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>36</td>\n",
       "      <td>50000</td>\n",
       "      <td>994</td>\n",
       "      <td>13</td>\n",
       "      <td>185</td>\n",
       "      <td>2</td>\n",
       "      <td>105</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>40646</th>\n",
       "      <td>8</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>25</td>\n",
       "      <td>4000</td>\n",
       "      <td>41</td>\n",
       "      <td>5</td>\n",
       "      <td>100</td>\n",
       "      <td>2</td>\n",
       "      <td>93</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       job  marital  education  targeted  default  housing  loan  contact  \\\n",
       "40829    0        2          1         1        0        1     0        0   \n",
       "29715    9        1          1         1        0        0     1        0   \n",
       "35015    9        1          1         1        0        1     0        0   \n",
       "41008    0        2          2         0        0        0     0        0   \n",
       "40646    8        2          1         1        0        1     0        1   \n",
       "\n",
       "       month  poutcome  age  salary  balance  day  duration  campaign  pdays  \\\n",
       "40829      1         2   40   50000      100   11       221         1    461   \n",
       "29715      3         0   45   60000      366    3       235         2    169   \n",
       "35015      8         0   32   60000     -360    6       131         2    344   \n",
       "41008      1         2   36   50000      994   13       185         2    105   \n",
       "40646      1         0   25    4000       41    5       100         2     93   \n",
       "\n",
       "       previous  \n",
       "40829         1  \n",
       "29715        12  \n",
       "35015         2  \n",
       "41008         3  \n",
       "40646         2  "
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import statsmodels.api as sm\n",
    "X_train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>const</th>\n",
       "      <th>default</th>\n",
       "      <th>housing</th>\n",
       "      <th>loan</th>\n",
       "      <th>contact</th>\n",
       "      <th>poutcome</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>40829</th>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>29715</th>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>35015</th>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>41008</th>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>40646</th>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       const  default  housing  loan  contact  poutcome\n",
       "40829    1.0        0        1     0        0         2\n",
       "29715    1.0        0        0     1        0         0\n",
       "35015    1.0        0        1     0        0         0\n",
       "41008    1.0        0        0     0        0         2\n",
       "40646    1.0        0        1     0        1         0"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train_sm = sm.add_constant(X_train[cols])\n",
    "X_train_sm.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<table class=\"simpletable\">\n",
       "<caption>OLS Regression Results</caption>\n",
       "<tr>\n",
       "  <th>Dep. Variable:</th>        <td>response</td>     <th>  R-squared:         </th> <td>   0.223</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.222</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   378.4</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Date:</th>             <td>Sun, 01 Nov 2020</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Time:</th>                 <td>10:07:42</td>     <th>  Log-Likelihood:    </th> <td> -2842.3</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>No. Observations:</th>      <td>  6605</td>      <th>  AIC:               </th> <td>   5697.</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Df Residuals:</th>          <td>  6599</td>      <th>  BIC:               </th> <td>   5737.</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Df Model:</th>              <td>     5</td>      <th>                     </th>     <td> </td>   \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   \n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "      <td></td>        <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>const</th>    <td>    0.2556</td> <td>    0.010</td> <td>   26.881</td> <td> 0.000</td> <td>    0.237</td> <td>    0.274</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>default</th>  <td>   -0.0629</td> <td>    0.056</td> <td>   -1.127</td> <td> 0.260</td> <td>   -0.172</td> <td>    0.047</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>housing</th>  <td>   -0.1904</td> <td>    0.010</td> <td>  -19.192</td> <td> 0.000</td> <td>   -0.210</td> <td>   -0.171</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>loan</th>     <td>   -0.0730</td> <td>    0.014</td> <td>   -5.394</td> <td> 0.000</td> <td>   -0.099</td> <td>   -0.046</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>contact</th>  <td>   -0.0618</td> <td>    0.014</td> <td>   -4.346</td> <td> 0.000</td> <td>   -0.090</td> <td>   -0.034</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>poutcome</th> <td>    0.1873</td> <td>    0.006</td> <td>   30.507</td> <td> 0.000</td> <td>    0.175</td> <td>    0.199</td>\n",
       "</tr>\n",
       "</table>\n",
       "<table class=\"simpletable\">\n",
       "<tr>\n",
       "  <th>Omnibus:</th>       <td>750.000</td> <th>  Durbin-Watson:     </th> <td>   2.039</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>1028.279</td> \n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Skew:</th>          <td> 0.954</td>  <th>  Prob(JB):          </th> <td>5.15e-224</td>\n",
       "</tr>\n",
       "<tr>\n",
       "  <th>Kurtosis:</th>      <td> 3.306</td>  <th>  Cond. No.          </th> <td>    16.9</td> \n",
       "</tr>\n",
       "</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified."
      ],
      "text/plain": [
       "<class 'statsmodels.iolib.summary.Summary'>\n",
       "\"\"\"\n",
       "                            OLS Regression Results                            \n",
       "==============================================================================\n",
       "Dep. Variable:               response   R-squared:                       0.223\n",
       "Model:                            OLS   Adj. R-squared:                  0.222\n",
       "Method:                 Least Squares   F-statistic:                     378.4\n",
       "Date:                Sun, 01 Nov 2020   Prob (F-statistic):               0.00\n",
       "Time:                        10:07:42   Log-Likelihood:                -2842.3\n",
       "No. Observations:                6605   AIC:                             5697.\n",
       "Df Residuals:                    6599   BIC:                             5737.\n",
       "Df Model:                           5                                         \n",
       "Covariance Type:            nonrobust                                         \n",
       "==============================================================================\n",
       "                 coef    std err          t      P>|t|      [0.025      0.975]\n",
       "------------------------------------------------------------------------------\n",
       "const          0.2556      0.010     26.881      0.000       0.237       0.274\n",
       "default       -0.0629      0.056     -1.127      0.260      -0.172       0.047\n",
       "housing       -0.1904      0.010    -19.192      0.000      -0.210      -0.171\n",
       "loan          -0.0730      0.014     -5.394      0.000      -0.099      -0.046\n",
       "contact       -0.0618      0.014     -4.346      0.000      -0.090      -0.034\n",
       "poutcome       0.1873      0.006     30.507      0.000       0.175       0.199\n",
       "==============================================================================\n",
       "Omnibus:                      750.000   Durbin-Watson:                   2.039\n",
       "Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1028.279\n",
       "Skew:                           0.954   Prob(JB):                    5.15e-224\n",
       "Kurtosis:                       3.306   Cond. No.                         16.9\n",
       "==============================================================================\n",
       "\n",
       "Warnings:\n",
       "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n",
       "\"\"\""
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lr1 = sm.OLS(y_train, X_train_sm).fit()\n",
    "lr1.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Features</th>\n",
       "      <th>VIF</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>age</td>\n",
       "      <td>11.79</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>education</td>\n",
       "      <td>6.43</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>targeted</td>\n",
       "      <td>6.43</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>salary</td>\n",
       "      <td>5.38</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>pdays</td>\n",
       "      <td>5.21</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>marital</td>\n",
       "      <td>4.35</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>day</td>\n",
       "      <td>4.10</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>month</td>\n",
       "      <td>3.60</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>housing</td>\n",
       "      <td>3.14</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>campaign</td>\n",
       "      <td>2.87</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>job</td>\n",
       "      <td>2.85</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>duration</td>\n",
       "      <td>2.26</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>previous</td>\n",
       "      <td>1.91</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>poutcome</td>\n",
       "      <td>1.76</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>balance</td>\n",
       "      <td>1.30</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>loan</td>\n",
       "      <td>1.20</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>contact</td>\n",
       "      <td>1.15</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>default</td>\n",
       "      <td>1.02</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     Features    VIF\n",
       "10        age  11.79\n",
       "2   education   6.43\n",
       "3    targeted   6.43\n",
       "11     salary   5.38\n",
       "16      pdays   5.21\n",
       "1     marital   4.35\n",
       "13        day   4.10\n",
       "8       month   3.60\n",
       "5     housing   3.14\n",
       "15   campaign   2.87\n",
       "0         job   2.85\n",
       "14   duration   2.26\n",
       "17   previous   1.91\n",
       "9    poutcome   1.76\n",
       "12    balance   1.30\n",
       "6        loan   1.20\n",
       "7     contact   1.15\n",
       "4     default   1.02"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from statsmodels.stats.outliers_influence import variance_inflation_factor\n",
    "# Create a dataframe that will contain the names of all the feature variables and their respective VIFs\n",
    "vif = pd.DataFrame()\n",
    "vif['Features'] = X_train.columns\n",
    "vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]\n",
    "vif['VIF'] = round(vif['VIF'], 2)\n",
    "vif = vif.sort_values(by = \"VIF\", ascending = False)\n",
    "vif"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,\n",
       "                       criterion='gini', max_depth=5, max_features='auto',\n",
       "                       max_leaf_nodes=50, max_samples=None,\n",
       "                       min_impurity_decrease=0.0, min_impurity_split=None,\n",
       "                       min_samples_leaf=1, min_samples_split=2,\n",
       "                       min_weight_fraction_leaf=0.0, n_estimators=100,\n",
       "                       n_jobs=None, oob_score=False, random_state=42, verbose=0,\n",
       "                       warm_start=False)"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "rfc = RandomForestClassifier(max_depth=5, random_state=42,max_leaf_nodes=50)\n",
    "rfc.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8392127176381529"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cv1_score= cross_val_score(rfc,X_train,y_train, cv=5)\n",
    "np.mean(cv1_score)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.86      0.97      0.91      1279\n",
      "           1       0.80      0.45      0.57       373\n",
      "\n",
      "    accuracy                           0.85      1652\n",
      "   macro avg       0.83      0.71      0.74      1652\n",
      "weighted avg       0.84      0.85      0.83      1652\n",
      "\n"
     ]
    }
   ],
   "source": [
    "y_pred1 = rfc.predict(X_test)\n",
    "print(classification_report(y_test, y_pred1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.5728987993138936"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "f1_score(y_test,y_pred1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1236,   43],\n",
       "       [ 206,  167]], dtype=int64)"
      ]
     },
     "execution_count": 47,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "confusion_matrix(y_test,y_pred1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7070505819937242"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import roc_auc_score\n",
    "roc_auc_score(y_test,y_pred1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RFE(estimator=RandomForestClassifier(bootstrap=True, ccp_alpha=0.0,\n",
       "                                     class_weight=None, criterion='gini',\n",
       "                                     max_depth=5, max_features='auto',\n",
       "                                     max_leaf_nodes=50, max_samples=None,\n",
       "                                     min_impurity_decrease=0.0,\n",
       "                                     min_impurity_split=None,\n",
       "                                     min_samples_leaf=1, min_samples_split=2,\n",
       "                                     min_weight_fraction_leaf=0.0,\n",
       "                                     n_estimators=100, n_jobs=None,\n",
       "                                     oob_score=False, random_state=42,\n",
       "                                     verbose=0, warm_start=False),\n",
       "    n_features_to_select=5, step=1, verbose=0)"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.feature_selection import RFE\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "scaler = MinMaxScaler()\n",
    "rfe1 = RFE(rfc, 5)\n",
    "rfe1.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([False, False, False, False, False,  True, False, False,  True,\n",
       "        True, False, False, False, False,  True, False,  True, False])"
      ]
     },
     "execution_count": 50,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rfe1.support_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['housing', 'month', 'poutcome', 'duration', 'pdays'], dtype='object')"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train.columns[rfe1.support_]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,\n",
       "                       criterion='gini', max_depth=5, max_features='auto',\n",
       "                       max_leaf_nodes=50, max_samples=None,\n",
       "                       min_impurity_decrease=0.0, min_impurity_split=None,\n",
       "                       min_samples_leaf=1, min_samples_split=2,\n",
       "                       min_weight_fraction_leaf=0.0, n_estimators=100,\n",
       "                       n_jobs=None, oob_score=False, random_state=42, verbose=0,\n",
       "                       warm_start=False)"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cols = X_train.columns[rfe1.support_]\n",
    "rfc.fit(X_train[cols],y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.631911532385466"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred3 = rfc.predict(X_test[cols])\n",
    "f1_score(y_pred3,y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[1219,  173],\n",
       "       [  60,  200]], dtype=int64)"
      ]
     },
     "execution_count": 54,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "confusion_matrix(y_pred3,y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
