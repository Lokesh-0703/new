
class Employee:
    def __init__(self, name, salary):
        self.__name = name  
        self.__salary = salary  

    
    def get_name(self):
        return self.__name

    
    def set_name(self, name):
        self.__name = name

    def get_salary(self):
        return self.__salary

    def set_salary(self, salary):
        if salary >= 0:
            self.__salary = salary
        else:
            print("Invalid salary")

    def __add__(self, other):
        return self.__salary + other.__salary

class Manager(Employee):
    def __init__(self, name, salary, department):
        super().__init__(name, salary)  
        self.department = department

    def display_info(self, show_department=True):
        print("Name:", self.get_name())
        print("Salary:", self.get_salary())
        if show_department:
            print("Department:", self.department)

def main():
    emp1 = Employee("Alice", 50000)
    emp2 = Employee("Bob", 60000)
    print("Initial salary of emp1:", emp1.get_salary())
    emp1.set_salary(55000)
    print("Updated salary of emp1:", emp1.get_salary())
    total_salary = emp1 + emp2
    print("Total salary of emp1 and emp2:", total_salary)
    mgr = Manager("Charlie", 80000, "IT")
    print("\nManager Info:")
    mgr.display_info()
    print("\nManager Info (without department):")
    mgr.display_info(show_department=False)

if __name__ == "__main__":
    main()