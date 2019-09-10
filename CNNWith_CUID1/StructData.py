class StructData:
    paramX=[]
    paramY=[]
    paramZ=[]
    paramGX=[]
    paramGY=[]
    paramGZ=[]
    paramFeatures=[]
    ParamAnswers_raw=[]

    def __init__(self,x,y,z,gx,gy,gz,features,answers):
        self.paramX=x
        self.paramY=y
        self.paramZ=z
        self.paramGX=gx
        self.paramGY=gy
        self.paramGZ=gz
        self.paramFeatures=features
        self.ParamAnswers_raw=answers

    def get_paramX(self):
        return paramX
    def get_paramY(self):
        return paramY
    def get_paramZ(self):
        return paramZ
    def get_paramGX(self):
        return paramGX
    def get_paramGY(self):
        return paramGY
    def get_paramGZ(self):
        return paramGZ
    def get_paramFeatures(self):
        return paramFeatures
    def get_ParamAnswers_raw(self):
        return ParamAnswers_raw