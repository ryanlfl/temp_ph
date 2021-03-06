SELECT AL2.StorerKey, 
AL2.Facility, 
AL2.AddDate, 
AL2.EditDate, 
AL2.Type, 
AL2.ExternOrderKey, 
AL2.OrderKey, 
AL2.OrderDate, 
AL1.LoadKey, 
AL1.MBOLKey, 
AL2.DeliveryDate, 
AL2.Status, 
AL2.SOStatus, 
AL2.Route, 
AL2.BillToKey, 
AL2.B_Company, 
AL3.B_Address1, 
AL2.B_Address1, 
AL2.B_Address3, 
AL2.B_City, 
AL2.ConsigneeKey, 
AL2.C_Company, 
AL2.C_Address1, 
AL2.C_Address2, 
AL2.C_Address3, 
AL2.C_City, 
AL3.Company, 
AL3.Address1, 
AL3.Address2, 
AL3.Address3, 
AL3.City, 
SUM ( AL1.OpenQty ) AS 'Open Quantity',
SUM ( case when AL6.CaseCnt = 0 then 0 else AL1.OpenQty / AL6.CaseCnt end ) AS 'Open Cases', 
sum (  AL1.OriginalQty  ) AS 'Original Quantity', 
SUM ( case when  AL6.CaseCnt  = 0 then 0 else ( AL1.QtyAllocated + AL1.QtyPicked + AL1.ShippedQty) / AL6.CaseCnt end ) AS 'New Cases', 
SUM ( AL1.QtyAllocated  +  AL1.QtyPicked  + AL1.ShippedQty) AS 'New Quantity', 
SUM ( case when  AL6.CaseCnt  = 0 then 0 else ( AL1.QtyAllocated + AL1.QtyPicked + AL1.ShippedQty) / AL6.CaseCnt end ) AS 'New Cases 2', 
SUM ( AL4.STDGROSSWGT * AL1.OriginalQty ) AS 'Original Weight', 
SUM( AL4.STDGROSSWGT *  ( AL1.QtyAllocated + AL1.QtyPicked + AL1.ShippedQty ) ) AS 'New Weight', 
SUM ( AL4.STDCUBE * AL1.OriginalQty )  AS 'Original Cube', 
SUM( AL4.STDCUBE *  ( AL1.QtyAllocated + AL1.QtyPicked  + AL1.ShippedQty)  ) AS 'New Cube', 
AL2.UserDefine06, 
convert ( char ( 60 ) ,AL2.Notes ) AS 'Notes', 
convert ( char ( 60 ) ,AL2.Notes2 ) AS 'Notes2', 
AL2.OrderGroup, 
AL2.Priority, 
AL2.UserDefine01, 
AL2.UserDefine02, 
AL2.UserDefine03, 
AL2.UserDefine04, 
AL2.BuyerPO, 
AL2.EditWho, 
AL2.UserDefine06, 
AL2.InvoiceNo, 
AL2.ExternPOKey, 
AL3.CustomerGroupCode, 
AL2.Door, 
AL2.Stop, 
AL2.ContainerType, 
AL4.SUSR3, 
AL5.Description, 
AL2.UserDefine09, 
SUM ( case when AL6.InnerPack = 0 then 0 else AL1.OpenQty / AL6.InnerPack end ) AS 'Open Pack', 
SUM ( case when  AL6.InnerPack  = 0 then 0 else ( AL1.QtyAllocated + AL1.QtyPicked + AL1.ShippedQty) / AL6.InnerPack end ) AS 'New Pack', 
SUM ( case when  AL6.InnerPack  = 0 then 0 else ( AL1.QtyAllocated + AL1.QtyPicked + AL1.ShippedQty) / AL6.InnerPack end ) AS 'New Pack 2' 

FROM dbo.V_ORDERS AL2, 
dbo.V_STORER AL3, 
dbo.V_ORDERDETAIL AL1 

LEFT OUTER JOIN dbo.V_SKU AL4 
ON (AL1.Sku=AL4.Sku AND AL1.StorerKey=AL4.StorerKey) 

LEFT OUTER JOIN dbo.V_CODELKUP AL5 
ON (AL4.SUSR3=AL5.Code) 

LEFT OUTER JOIN dbo.V_PACK AL6 
ON (AL4.PACKKey=AL6.PackKey) 

WHERE (AL2.OrderKey=AL1.OrderKey AND AL3.StorerKey=AL2.StorerKey)
AND AL2.DeliveryDate BETWEEN DATEADD(DD,-14,GETDATE()) AND GETDATE()
--AND ((AL2.StorerKey='IDSMED' AND AL2.Facility='CDC1' 
--AND ((AL2.StorerKey='RC01' AND AL2.Facility='RC01' 
--AND ((AL2.StorerKey='FBP' AND AL2.Facility='CDC1' 
--AND ((AL2.StorerKey='DPI' AND AL2.Facility='CDC1' 
AND ((AL2.StorerKey='NIKEPH' AND AL2.Facility='CDC1' 
AND 1=1 
--AND AL2.Status IN ('0', '1', '2', '3', '5')
AND AL2.Status IN ('9') 
AND (AL5.LISTNAME='PRINCIPAL' OR AL5.LISTNAME IS NULL) AND 1=1 AND 1=1)) 

GROUP BY AL2.StorerKey, 
AL2.Facility, 
AL2.AddDate, 
AL2.EditDate, 
AL2.Type, 
AL2.ExternOrderKey, 
AL2.OrderKey, 
AL2.OrderDate, 
AL1.LoadKey, 
AL1.MBOLKey, 
AL2.DeliveryDate, 
AL2.Status, 
AL2.SOStatus, 
AL2.Route, 
AL2.BillToKey, 
AL2.B_Company, 
AL3.B_Address1, 
AL2.B_Address1, 
AL2.B_Address3, 
AL2.B_City, 
AL2.ConsigneeKey, 
AL2.C_Company, 
AL2.C_Address1, 
AL2.C_Address2, 
AL2.C_Address3, 
AL2.C_City, 
AL3.Company, 
AL3.Address1, 
AL3.Address2, 
AL3.Address3, 
AL3.City, 
AL2.UserDefine06, 
convert ( char ( 60 ) ,AL2.Notes ) , 
convert ( char ( 60 ) ,AL2.Notes2 ) , 
AL2.OrderGroup, 
AL2.Priority, 
AL2.UserDefine01, 
AL2.UserDefine02, 
AL2.UserDefine03, 
AL2.UserDefine04, 
AL2.BuyerPO, 
AL2.EditWho, 
AL2.UserDefine06, 
AL2.InvoiceNo, 
AL2.ExternPOKey, 
AL3.CustomerGroupCode, 
AL2.Door, 
AL2.Stop, 
AL2.ContainerType, 
AL4.SUSR3, 
AL5.Description, 
AL2.UserDefine09
