class Animal(object):
    def __init__(self,  name, age, sex):
        self.__name = name
        self.__age = age
        self.__sex = sex

    def __str__(self):
        return "name = " + self.__name + " age = " + self.__age + " sex =" + self.__sex

a = Animal('dog', '10', 'man')
print(a)