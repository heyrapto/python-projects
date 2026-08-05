# define an input
# take one value
# eval it
# print the value and log it

while True:
    i = input("Calculate anything: ");
    if i.lower() == "e":
        print("Exiting program, goodbye!");
        break;
    result = eval(i);
    print(result);
    print("Enter E to exit this program");
