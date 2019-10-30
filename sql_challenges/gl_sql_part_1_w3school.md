# Challenge Set 9
## Part I: W3Schools SQL Lab 

*Introductory level SQL*

--

This challenge uses the [W3Schools SQL playground](http://www.w3schools.com/sql/trysql.asp?filename=trysql_select_all). Please add solutions to this markdown file and submit.

0. Resources:
Some resources I used for this exercise
- creating a view [w3schools](https://www.w3schools.com/sql/sql_view.asp)
- what are joins [w3schools](https://www.w3schools.com/sql/sql_join.asp)
- how to join 3 tables [site](https://www.dofactory.com/sql/join)
- why is there a bracket around table name e.g. FROM [table][stack overflow](https://stackoverflow.com/questions/9917196/meaning-of-square-brackets-in-ms-sql-table-designer)

- creating markdown tables [site](https://www.tablesgenerator.com/markdown_tables)




1. Which customers are from the UK?
```
SELECT CustomerID, CustomerName, ContactName
FROM Customers
WHERE Country='UK'
```
    
|CustomerID|CustomerName|ContactName|
|----------|-----------|-----------|
|4	|Around the Horn|	Thomas Hardy|
|11	|B's Beverages	|Victoria Ashworth|
|16	|Consolidated Holdings	|Elizabeth Brown|
|19	|Eastern Connection	|Ann Devon|
|38	|Island Trading	|Helen Bennett|
|53	|North/South	|Simon Crowther|
|72	|Seven Seas Imports	|Hari Kumar|

2. What is the name of the customer who has the most orders?
```sql
CREATE VIEW cust_max_orders AS
SELECT CustomerID, COUNT(CustomerID)
FROM [Orders]
GROUP BY CustomerID
ORDER BY order_count DESC;

SELECT CustomerID, MAX(order_count) FROM [cust_max_orders];

SELECT * FROM [Customers] WHERE CustomerID=20;

```
- ANSWER: Ernst Handel


3. Which supplier has the highest average product price?
```sql
---table Products: ProductID	ProductName	SupplierID	CategoryID	Unit	Price

---table Suppliers: SupplierID	SupplierName	ContactName	Address	City	PostalCode	Country	Phone

SELECT SupplierID, MAX(avg_prices) FROM 
(
    SELECT SupplierID, AVG(Price) as avg_prices
    FROM [Products]
    GROUP BY SupplierID
    ORDER BY avg_prices DESC
);

SELECT SupplierID, SupplierName
FROM [Suppliers]
WHERE SupplierID = 18;
```

|SupplierID|	SupplierName|
|----|---|
|18	|Aux joyeux ecclésiastiques|



4. How many different countries are all the customers from? (*Hint:* consider [DISTINCT](http://www.w3schools.com/sql/sql_distinct.asp).)

```sql
SELECT COUNT(DISTINCT(Country)) FROM [Customers];
```
- ANSWER: 18 distinct countries



5. What category appears in the most orders?
```sql
--- OrderDetails: OrderDetailID	OrderID	ProductID	Quantity
--- Categories: CategoryID	CategoryName	Description
--- Products: ProductID	ProductName	SupplierID	CategoryID	Unit	Price

--- Plan is to join these three tables together, and then aggregate on the findal table to see which category appears most often. OrderDetails has 518 records, so the final joined table should have 518 record as well

SELECT CategoryID, CategoryName, MAX(cat_count) FROM
(
    SELECT Categories.CategoryID, Categories.CategoryName, COUNT(Categories.CategoryID) as cat_count
    FROM OrderDetails
    LEFT JOIN Products
        ON OrderDetails.ProductID=Products.ProductID
    LEFT JOIN Categories
        ON Products.CategoryID=Categories.CategoryID
    GROUP BY Categories.CategoryID
    ORDER BY cat_count DESC
)

```
|CategoryID	|CategoryName|	MAX(cat_count)|
|---|---|---|
|4|	Dairy Products	|100|


6. What was the total cost for each order?

```sql
SELECT OrderID, OrderDetails.ProductID, ProductName, Price, Quantity, (Price * Quantity) as total_rev
FROM OrderDetails
LEFT JOIN Products
    ON OrderDetails.ProductID=Products.ProductID
ORDER BY OrderID DESC
```


7. Which employee made the most sales (by total price)?
```sql
--- Employees: EmployeeID	LastName	FirstName	BirthDate	Photo	Notes

--- using query from previous question as a view
CREATE VIEW order_totals AS
SELECT OrderID, OrderDetails.ProductID, ProductName, Price, Quantity, (Price * Quantity) as total_rev
FROM OrderDetails
LEFT JOIN Products
    ON OrderDetails.ProductID=Products.ProductID
ORDER BY OrderID DESC

SELECT
  order_totals.OrderID,
  Employees.EmployeeID,
  Employees.LastName,
  Employees.FirstName,
  SUM(order_totals.total_rev) as total_price_sales
FROM Orders
LEFT JOIN order_totals
    ON Orders.OrderID = order_totals.OrderID
LEFT JOIN Employees
    ON Orders.EmployeeID = Employees.EmployeeID
GROUP BY Employees.EmployeeID
ORDER BY total_price_sales DESC;

```

|EmployeeID|	LastName|	FirstName|	total_price_sales|
|---|---|---|---|
|10250|	4|	Peacock	Margaret|	105696.49999999999|
|10258|	1|	Davolio	Nancy|	57690.38999999999|


8. Which employees have BS degrees? (*Hint:* look at the [LIKE](http://www.w3schools.com/sql/sql_like.asp) operator.)
```sql
SELECT EmployeeID, LastName, FirstName, Notes
FROM Employees
WHERE Notes LIKE '% BS%'
```

|EmployeeID|	LastName|	FirstName|	Notes|
|---|---|---|---|
|3|	Leverling|	Janet|	Janet has a BS degree in chemistry from Boston College). She has also completed a certificate program in food retailing management. Janet was hired as a sales associate and was promoted to sales representative.|
|5|	Buchanan|	Steven|	Steven Buchanan graduated from St. Andrews University, Scotland, with a BSC degree. Upon joining the company as a sales representative, he spent 6 months in an orientation program at the Seattle office and then returned to his permanent post in London, where he was promoted to sales manager. Mr. Buchanan has completed the courses 'Successful Telemarketing' and 'International Sales Management'. He is fluent in French.|




9. Which supplier of three or more products has the highest average product price? (*Hint:* look at the [HAVING](http://www.w3schools.com/sql/sql_having.asp) operator.)

```sql
---Suppliers: SupplierID	SupplierName	ContactName	Address	City	PostalCode	Country	Phone

CREATE VIEW suppliers_multi AS
SELECT SupplierID, COUNT(ProductID)
FROM [Products]
GROUP BY SupplierID
HAVING COUNT(ProductID) >= 3

SELECT suppliers_multi.SupplierID, Suppliers.SupplierName, Suppliers.Country, Suppliers.Phone, AVG(Products.Price) as avg_price
FROM suppliers_multi
INNER JOIN Products
    ON suppliers_multi.SupplierID=Products.SupplierID
INNER JOIN Suppliers
    ON suppliers_multi.SupplierID=Suppliers.SupplierID
GROUP BY suppliers_multi.SupplierID
ORDER BY avg_price DESC

```

|SupplierID|	SupplierName|	Country|	Phone|	avg_price|
|---|---|---|---|---|
|4|	Tokyo Traders|	Japan|	(03) 3555-5011|	46|
