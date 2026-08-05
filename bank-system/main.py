# view bal, withdraw, deposit, view transaction history

class BankAccount:
    def __init__(self, account_balance: int, account_owner: str):
        self.owner = account_owner
        self.transaction_history = []
        self._balance = account_balance

    # get balance
    def get_balance(self):
        return f"Your balance is: {self._balance}"

    # deposit funds
    def deposit(self):
        amount = int(input("How much do you wish to deposit? "))
        new_balance = self._balance + amount
        self.transaction_history.append(
            f"A sum of {amount} was deposited into your account"
        )
        return f"Deposit succesful, your new balance is: {new_balance}"

    # withdraw funds
    def withdraw(self):
        amount = int(input("How much do you wish to withdraw? "))
        if amount > self._balance:
            print("Insufficient funds.")
        else:
            new_balance = self._balance - amount
            self.transaction_history.append(
                f"A sum of {amount} was withdrawn from your account"
            )
            return f"Withdrawal succesful, your new balance is: {new_balance}"

    # view trans history
    def get_history(self):
        return self.transaction_history.sort()

    def run(self):
        while True:
            print("\n--- Menu ---")
            print("1. View Balance")
            print("2. Withdraw")
            print("3. Deposit")
            print("4. Transaction History")
            choice = input("Select an option (1-4). ")

            if choice == "1":
                balance = self.get_balance()
                print(balance)
            elif choice == "2":
                result = self.withdraw()
                print(result)
            elif choice == "3":
                result = self.deposit()
                print(result)
            elif choice == "4":
                result = self.get_history()
                print(result)
            else:
                print("Invalid input. Enter an option from 1-4.")


bank = BankAccount(100000000, "Caleb")
bank.run()
