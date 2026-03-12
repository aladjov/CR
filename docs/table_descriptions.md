SPS Tables Description

prod_corp_snowflake_provisioning_shared.salesforce.account:
1. ACCOUNT_ID (string): Unique identifier for the account.
2. CREATED_BY_EMPLOYEE_ID (string): Identifier for the employee who created the account.
3. LAST_MODIFIED_BY_EMPLOYEE_ID (string): Identifier for the employee who last modified the account.
4. CREATED_DATE (timestamp): Date and time when the account was created.
5. LAST_MODIFIED_DATE (timestamp): Date and time when the account was last modified.
6. ACCOUNT_NAME (string): Name of the account.
7. NAV_CUSTOMER_NUMBER (string): NAV customer number associated with the account.
8. TOP_PARENT_ACCOUNT_ID (string): Identifier of the top parent account in the hierarchy.
9. TOP_PARENT_ACCOUNT_NAME (string): Name of the top parent account.
10. HUB_ID (string): Identifier for the hub associated with the account.
11. NAV_CUSTOMER_NUMBER_2 (string): Secondary NAV customer number.
12. NAV_CUSTOMER_NUMBER_ACQUISITION (string): NAV customer number related to acquisitions.
13. NAV_CUSTOMER_NUMBER_EDIFICE (string): NAV customer number related to edifice.
14. MONTHLY_FEE_TYPE (string): Type of monthly fee applied to the account.
15. ERP_ACCOUNTING_APPLICATION (string): Accounting application used in ERP system.
16. ADAPTER (string): Adapter associated with the account.
17. FI_TEAM (string): Financial team responsible for the account.
18. CURRENT_CUSTOMER_SUCCESS_SEGMENT (string): Current segment in customer success.
19. CURRENT_CUSTOMER_SUCCESS_SUBSEGMENT (string): Current sub-segment in customer success.
20. TERRITORY_COUNTRY (string): Country of the account's territory.
21. CS_ADAPTER (string): Customer success adapter.
22. CS_ERP_ACCOUNTING_APPLICATION (string): ERP accounting application for customer success.
23. CUSTOMER_ENGAGEMENT_PLAN (string): Engagement plan for the customer.
24. EXISTING_CUSTOMER_CURRENT_FISCAL_YEAR (decimal(10,0)): Status of existing customer for the current fiscal year.
25. EXISTING_CUSTOMER_PREVIOUS_FISCAL_YEAR (decimal(10,0)): Status of existing customer for the previous fiscal year.
26. EXISTING_CUSTOMER_SECOND_PREVIOUS_FISCAL_YEAR (decimal(10,0)): Status of existing customer for the second previous fiscal year.
27. ANNUAL_REVENUE (decimal(38,0)): Annual revenue of the account.
28. ACCOUNT_TYPE (string): Type of the account.
29. ANNUAL_REVENUE_RANGE (string): Revenue range of the account on an annual basis.
30. INDUSTRY_SPS (string): Industry sector (SPS) the account belongs to.
31. FIRST_SUBSCRIPTION_DATE (timestamp): Date when the first subscription was made.
32. NET_NEW_CUSTOMER_DATE (timestamp): Date when the customer was considered net new.
33. ETL_CREATED_TIMESTAMP (timestamp): Timestamp when the record was initially created in the ETL process.
34. ETL_UPDATED_TIMESTAMP (timestamp): Timestamp when the record was last updated in the ETL process.
35. TERRITORY_RDM (string): Territory RDM (Regional Distribution Manager).
36. SALES_TEAM (string): Sales team responsible for the account.
37. ACCOUNT_ALSO_KNOWN_AS (string): Alternative name for the account.
38. WEBSITE (string): Website of the account.
39. BILLING_STREET (string): Billing street address of the account.
40. BILLING_CITY (string): Billing city of the account.
41. BILLING_COUNTRY (string): Billing country of the account.
42. BILLING_STATE (string): Billing state of the account.
43. BILLING_POSTAL_CODE (string): Billing postal code of the account.
44. ACCOUNT_LEVEL (string): Level classification of the account.
45. ACCOUNT_NOTES (string): Additional notes about the account.
46. ACCOUNT_NUMBER (string): Account number.
47. ANNIVERSARY_DATE__C (timestamp): Anniversary date.
48. ASSORTMENT_COMPANY_ID (string): Identifier for the assortment company associated with the account.
49. BILLING_LATITUDE (decimal(38,20)): Latitude of the billing address.
50. BILLING_LONGITUDE (decimal(38,20)): Longitude of the billing address.

prod_corp_snowflake_provisioning_shared.salesforce.case
1. CASE_ID (string): Unique identifier for the case.
2. CREATED_BY_EMPLOYEE_ID (string): Identifier of the employee who created the case.
3. LAST_MODIFIED_BY_EMPLOYEE_ID (string): Identifier of the employee who last modified the case.
4. CREATED_DATE (timestamp): Date and time when the case was created.
5. LAST_MODIFIED_DATE (timestamp): Date and time when the case was last modified.
6. ACCOUNT_ID (string): Identifier for the account associated with the case.
7. OWNER_EMPLOYEE_ID (string): Identifier of the employee who owns the case.
8. RELATED_TO_ACCOUNT_ID (string): Identifier for the account related to the case.
9. RECORD_TYPE_ID (string): Identifier for the type of record.
10. CLOSED_DATE (timestamp): Date and time when the case was closed.
11. FIRST_ASSIGNED_DATE_TIME (timestamp): Date and time when the case was first assigned.
12. FIRST_CLOSED_DATE_TIME (timestamp): Date and time when the case was first closed.
13. IMPLEMENTATION_PROJECT_ID (string): Identifier for the related implementation project.
14. CASE_NUMBER (string): The case number for reference.
15. ORIGIN (string): Origin of the case (e.g., email, phone).
16. CASE_REFERENCE (string): Reference associated with the case.
17. CASE_STATUS (string): Current status of the case.
18. SUB_REFERENCE (string): Additional reference information.
19. CASE_TYPE (string): Type of case (e.g., issue, query).
20. DETAIL_TAGS (string): Tags providing additional details about the case.
21. CASE_SUBJECT (string): Subject or title of the case.
22. PROJECT_STATUS_SNAPSHOT (string): Snapshot of the project status related to the case.
23. IS_CLOSED (decimal(1,0)): Indicates if the case is closed (1 for closed, 0 for open).
24. RESPONSE_DATE_TIME (timestamp): Date and time when a response was provided.
25. CASE_AGE_HOURS (decimal(10,0)): Age of the case in hours.
26. ECONOMIC_FACTORS_EXCEPTION (decimal(3,0)): Indicator for economic factors exception.
27. CANCEL_COMMENTS_DSAT (string): Comments regarding cancellation due to dissatisfaction.
28. CANCEL_COMMENTS_GENERAL (string): General comments on case cancellation.
29. CANCELLATION_REASON (string): Reason for case cancellation.
30. FUTURE_SERVICE_MANAGEMENT (string): Future considerations for service management.
31. FUTURE_SERVICE_MANAGEMENT_COMMENTS (string): Comments on future service management.
32. OUTCOME (string): Outcome of the case.
33. CONTACT_ID (string): Identifier for the contact associated with the case.
34. RECORD_TYPE_NAME (string): Name of the record type.
35. ETL_CREATED_TIMESTAMP (timestamp): Timestamp when the case record was created in the ETL system.
36. ETL_UPDATED_TIMESTAMP (timestamp): Timestamp when the case record was last updated in the ETL system.
37. REOPENED_DATE_TIME (timestamp): Date and time when the case was reopened.
38. ACCOUNT_CS_SEGMENT (string): Customer segmentation of the account.
39. TRADING_PARTNERS (string): Trading partners associated with the case.
40. CANCELLATION_REASON_CATEGORY (string): Category of the cancellation reason.
41. DESCRIPTION (string): Detailed description of the case.
42. PRIORITY_CHANGE_COMMENTS (string): Comments on the priority change of the case.
43. CASE_REOPEN_COUNT (decimal(18,0)): Count of times the case has been reopened.
44. ESCALATED_ISSUE_NUMBER (string): Identifier for the escalated issue.
45. SEVERITY (string): Severity level of the case.
46. CSAT_SURVEY_SENT (decimal(10,0)): Indicator if a customer satisfaction survey was sent (1 for sent, 0 for not sent).
47. CSAT_SURVEY_SENT_DATE (timestamp): Date and time when the customer satisfaction survey was sent.
48. ERP_ACCT_APP_POS (string): ERP account application position.
49. SPS_REQUESTOR_EMPLOYEE_ID (string): Identifier of the employee who requested SPS.
50. CUSTOMER_PHASE (string): Phase of the customer associated with the case.
Case Record Type = 'retention'
* pre cancellation
Case Record Type = 'cancellation' 
* post cancellation submission
Case Record Type = 'optimization'
* customer interventions
Case Record Type = 'Customer Ops Case'


prod_corp_snowflake_provisioning_shared.salesforce.case_history:
* ETL_CREATED_TIMESTAMP: timestamp (Timestamp indicating when the ETL process created this record)
* ETL_UPDATED_TIMESTAMP: timestamp (Timestamp indicating when the ETL process last updated this record)
* CASE_HISTORY_ID: string (Unique identifier for each case history record)
* IS_DELETED: decimal(10,0) (Flag indicating whether the record is deleted; typically 0 or 1)
* CASE_ID: string (Unique identifier for the related case)
* CREATED_BY_EMPLOYEE_ID: string (Identifier for the employee who created the case history record)
* CREATED_DATE: timestamp (Timestamp indicating when the case history record was created)
* FIELD: string (Name of the field that was changed in the case)
* OLD_VALUE: string (Previous value of the field before the change)
* NEW_VALUE: string (New value of the field after the change)

prod_corp_snowflake_provisioning_shared.salesforce.contract:
1. CONTRACT_ID (string): Unique identifier for the contract.
2. ACTIVE_SUBSCRIPTION_PRODUCT_SELL_GROUP (string): Identifies the active subscription product sell group.
3. CONTRACT_START_DATE (timestamp): The start date of the contract.
4. CONTRACT_END_DATE (timestamp): The end date of the contract.
5. CONTRACT_TERM_TYPE (string): The type of term the contract is under.
6. CONTRACT_TERM (string): The duration of the contract term.
7. ADVANCE_CANCEL_NOTICE (string): The advance notice required for cancellation.
8. DOCUMENT_PLAN_TYPE (string): The type of document plan associated with the contract.
9. DOCUMENT_PLAN (string): The specific document plan related to the contract.
10. DOCUMENT_ALLOTMENT (decimal(18,0)): The allotment of documents in the contract.
11. CONTRACT_STATUS (string): The current status of the contract.
12. CONTRACT_DIVISION (string): The division that the contract pertains to.
13. NEXT_RENEWAL_DATE (timestamp): The next date the contract is set to renew.
14. ETL_CREATED_TIMESTAMP (timestamp): The timestamp when the ETL process created this record.
15. ETL_UPDATED_TIMESTAMP (timestamp): The timestamp when the ETL process last updated this record.
16. CURRENT_TERM (decimal(28,0)): The current term of the contract.
17. BILLING_TERMINATION_DATE (timestamp): The date the billing for the contract terminates.
18. ACCOUNT_ID (string): The identifier for the account associated with the contract.
19. DESCRIPTION (string): A description of the contract.
20. CONTRACT_NAME (string): The name of the contract.
21. ACTIVATED_DATE (timestamp): The date the contract was activated.
22. CONTRACT_NUMBER (string): The number assigned to the contract.
It seems that we have more than one active contract per accountactive_contract_count,num_accounts
1,36133
2,5207
3,1179
4,350
5,150
6,61
7,28
8,15
9,13
10,1
11,5
12,1
13,1
21,1

prod_corp_snowflake_provisioning_shared.salesforce.implementation_project:
* IMPLEMENTATION_PROJECT_ID (string): A unique identifier for each implementation project.
* CREATED_BY_EMPLOYEE_ID (string): The identifier of the employee who created the project record.
* LAST_MODIFIED_BY_EMPLOYEE_ID (string): The identifier of the employee who last modified the project record.
* CREATED_DATE (timestamp): The date and time when the project record was created.
* LAST_MODIFIED_DATE (timestamp): The date and time when the project record was last updated.
* IMPLEMENTATION_PROJECT_NAME (string): The name given to the implementation project.
* PRODUCTION_DATE (date): The date when the project went into production.
* PRODUCTION_READY_DATE (date): The date when the project was ready for production.
* REPORTING_PRODUCTION_DATE (date): The date for reporting production status.
* ACCOUNT_ID (string): The identifier of the account associated with the project.
* RELATED_TO_ACCOUNT_ID (string): The identifier of an account related to the project.
* PROJECT_STATUS (string): The current status of the project (e.g., In Progress, Completed).
* OPPORTUNITY_ID (string): The identifier of the opportunity linked to the project.
* PROJECT_TYPE (string): The type or category of the project.
* RECORD_TYPE_ID (string): The type of record associated with the project.
* PARENT_RELEASE_MANAGEMENT_PROJECT (string): The identifier of the parent release management project, if any.
* PROJECT_START_DATE (timestamp): The date and time when the project started.
* OWNER_EMPLOYEE_ID (string): The employee identifier of the project owner.
* DOCUMENT (string): Any documentation associated with the project.
* ETL_CREATED_TIMESTAMP (timestamp): The timestamp when the ETL (Extract, Transform, Load) process created this record.
* ETL_UPDATED_TIMESTAMP (timestamp): The timestamp when the ETL process last updated this record.
* INITIAL_TARGETED_IMPLEMENTATION_DATE (timestamp): The initial target date for the project's implementation.
* SOW_TARGETED_IMPLEMENTATION_DATE (timestamp): The Statement of Work specified target date for implementation.
* TARGETED_IMPLEMENTATION_DATE (timestamp): The final target date for implementation.
* PROJECT_SCOPE (string): A description of the scope and objectives of the project.
* SUPPLIER_VAN (string): The supplier or vendor assigned to the project.
* IS_HYBRID (decimal(1,0)): Indicates if the project is of a hybrid nature (1 for Yes, 0 for No).
* LEAD_SOURCE (string): The source from which the project lead originated.
* VENDOR_NUMBER (string): The vendor number associated with the project.
* OPPORTUNITY_CAMPAIGN_SOURCE (string): The campaign source of the associated opportunity.
* EXEMPTED_DOCUMENTS (string): Documents that are exempted from the project requirements.
* NII_TEAM (string): The NII (Network Integration Initiative) team involved in the project.
* SPECIALTY_TEAM (string): The specialty team working on the project.
* PORTFOLIO (string): The portfolio to which the project belongs.
* IMPLEMENTATION_ANALYST_EMPLOYEE_ID (string): The employee identifier of the implementation analyst.
* BUSINESS_ANALYST_EMPLOYEE_ID (string): The employee identifier of the business analyst.
* PARENT_IMPLEMENTATION_PROJECT_ID (string): The identifier of the parent implementation project, if any.
* FI_TEAM_PICK_LIST (string): The pick list for the FI (Financial Integration) team.
* FI_TEAM (string): The FI team involved in the project.
* SECONDARY_RESOURCE_EMPLOYEE_ID (string): The employee identifier of a secondary resource.
* WORK_UNIT_TYPE (string): The type of work unit associated with the project.
* UNIQUE_WEBFORMS_SETUP (string): Indicates if there is a unique web form setup for the project.
* SOLUTION_STATUS (string): The current status of the project solution.
* PROJECT_STAGE (string): The stage of the project (e.g., Planning, Execution, Closing).
* PROJECT_MANAGER_EMPLOYEE_ID (string): The employee identifier of the project manager.
* SETUP_COMPLETE_DATE (timestamp): The date and time when the project setup was completed.
* ON_HOLD_DATE (timestamp): The date and time when the project was put on hold.
* CANCELLED_DATE (timestamp): The date and time when the project was canceled.
* ON_HOLD_REASON (string): The reason why the project was put on hold.
* DIRECT_EDI_MIGRATION_PROJECT (decimal(10,0)): Indicates if the project is a direct EDI (Electronic Data Interchange) migration project.


prod_corp_snowflake_provisioning_shared.salesforce.opportunity:
* OPPORTUNITY_ID (string): The unique identifier for the opportunity.
* OWNER_ID (string): Identifier for the owner of the opportunity.
* CREATED_BY_EMPLOYEE_ID (string): Identifier for the employee who created the opportunity.
* LAST_MODIFIED_BY_EMPLOYEE_ID (string): Identifier for the employee who last modified the opportunity.
* CREATED_DATE (timestamp): The date and time when the opportunity was created.
* LAST_MODIFIED_DATE (timestamp): The date and time when the opportunity was last modified.
* OPPORTUNITY_NAME (string): The name of the opportunity.
* OPPORTUNITY_STATUS (string): The current status of the opportunity.
* CAMPAIGN_ID (string): Identifier for the associated campaign.
* MONTHLY_FEE_CHANGE_TYPE_FROM (string): Original type of the monthly fee change.
* MONTHLY_FEE_CHANGE_TYPE_TO (string): New type of the monthly fee change.
* CURRENCY_KEY (decimal(10,0)): The currency key associated with the opportunity.
* ESTIMATED_CLOSE_DATE (date): The estimated date when the opportunity is expected to close.
* ESTIMATED_ONE_TIME_FEES (decimal(18,2)): The estimated one-time fees for the opportunity.
* ESTIMATED_NET_MONTHLY_RECURRING (decimal(18,2)): The estimated net monthly recurring revenue.
* SALE_MADE (decimal(3,0)): Indicates whether a sale was made (typically 1 for yes, 0 for no).
* SALE_DATE (date): The date when the sale was made.
* BOOKINGS_ONE_TIME_FEES (decimal(18,2)): The booked one-time fees.
* BOOKINGS_MONTHLY_RECURRING (decimal(18,2)): The booked monthly recurring revenue.
* USD_BOOKINGS_ONE_TIME_FEES (decimal(18,2)): The booked one-time fees in USD.
* USD_BOOKINGS_MONTHLY_RECURRING (decimal(18,2)): The booked monthly recurring revenue in USD.
* LIFT (decimal(18,2)): The lift value associated with the opportunity.
* ACCOUNT_ID (string): Identifier for the account associated with the opportunity.
* TASK_LAST_MODIFIED_TIMESTAMP (timestamp): Timestamp when the task was last modified.
* CLOSED (decimal(1,0)): Indicator of whether the opportunity is closed (typically 1 for yes, 0 for no).
* SUBMITTED_DATE (timestamp): The date and time when the opportunity was submitted.
* TESTING_FEE (decimal(18,2)): The testing fee associated with the opportunity.
* PRIMARY_PO_QUANTITY (decimal(15,0)): The primary purchase order quantity.
* PRIMARY_PO_VALUE (decimal(18,2)): The primary purchase order value.
* PO_VALUE_RANGE (string): The range of the purchase order value.
* PRIMARY_SKU_COUNT (decimal(10,0)): The count of primary stock keeping units (SKU).
* ESCALATION_DATE (date): The date when the opportunity was escalated.
* ESCALATION_RESOLUTION_DATE (date): The date when the escalation was resolved.
* COMPETITIVE_KILL (decimal(1,0)): Indicator for a competitive kill opportunity (typically 1 for yes, 0 for no).
* COMMISSIONABLE_ARR (decimal(18,2)): The commissionable annual recurring revenue.
* NON_COMMISSIONABLE_ARR (decimal(18,2)): The non-commissionable annual recurring revenue.
* USD_COMMISSIONABLE_ARR (decimal(18,2)): The commissionable annual recurring revenue in USD.
* USD_NON_COMMISSIONABLE_ARR (decimal(18,2)): The non-commissionable annual recurring revenue in USD.
* FIRST_SUBSCRIPTION_SALE (decimal(1,0)): Indicator for the first subscription sale (typically 1 for yes, 0 for no).
* RECORD_TYPE_ID (string): Identifier for the record type.
* COMP_KILL_MONTHLY_OPTION (string): The monthly option for a competitive kill.
* USD_ESTIMATED_ONE_TIME_FEES (decimal(18,2)): The estimated one-time fees in USD.
* USD_ESTIMATED_NET_MONTHLY_RECURRING (decimal(18,2)): The estimated net monthly recurring revenue in USD.
* IS_OPEN (decimal(1,0)): Indicator of whether the opportunity is open (typically 1 for yes, 0 for no).
* CLOSED_WON_DECISIONS (decimal(18,0)): The number of closed-won decisions.
* CONTRACT_EFFECTIVE_TIMING (string): The effective timing of the contract.
* CUSTOMER_ASK (string): The customer's ask or requirements.
* RATIONALE_FOR_APPROVAL (string): The rationale for the approval of the opportunity.
* ESTIMATED_MRR_REDUCTION_AMOUNT (decimal(38,0)): The estimated monthly recurring revenue reduction amount.
* PRIMARY_CONTACT_ID (string): Identifier for the primary contact associated with the opportunity.


prod_corp_snowflake_provisioning_shared.salesforce.opportunity_product:
1. OPPORTUNITY_PRODUCT_ID (string): A unique identifier for the opportunity product.
2. OPPORTUNITY_ID (string): A unique identifier for the related opportunity.
3. PRODUCT_ID (string): A unique identifier for the product.
4. MONTHLY_RECURRING (decimal(18,2)): The monthly recurring revenue amount for the product.
5. ADJUSTED_MONTHLY_RECURRING (decimal(18,2)): The adjusted monthly recurring revenue amount for the product.
6. ONE_TIME_FEES (decimal(18,2)): One-time fee amount associated with the product.
7. USD_MONTHLY_RECURRING (decimal(18,2)): Monthly recurring revenue amount in USD.
8. USD_ADJUSTED_MONTHLY_RECURRING (decimal(18,2)): Adjusted monthly recurring revenue amount in USD.
9. USD_ONE_TIME_FEES (decimal(18,2)): One-time fee amount in USD.
10. WIN_LOSS (string): Indicates whether the opportunity was won or lost.
11. RETAILER_ACCOUNT_ID (string): A unique identifier for the retailer account.
12. ETL_CREATED_TIMESTAMP (timestamp): Timestamp when the record was created by the ETL process.
13. ETL_UPDATED_TIMESTAMP (timestamp): Timestamp when the record was last updated by the ETL process.
14. OPPORTUNITY_PRODUCT_DESCRIPTION (string): Description of the opportunity product.
15. QUANTITY (decimal(10,2)): Quantity of the product.
16. TOTAL_PRICE (decimal(16,2)): Total price for the specified quantity of the product.
17. UNIT_PRICE (decimal(16,2)): Unit price of the product.
18. INVOICE_STATUS_REOCCURRING (string): Status of the reoccurring invoice.
19. BILLING_TERM (string): Billing term for the product.
20. IS_FLAT_RATE (decimal(10,0)): Indicates whether the pricing is a flat rate.
21. CREATED_DATE (timestamp): Date when the opportunity product record was created.
22. ANNUAL_RECURRING (decimal(18,2)): Annual recurring revenue amount for the product.
23. ADJUSTED_ANNUAL_RECURRING (decimal(18,2)): Adjusted annual recurring revenue amount for the product.
24. USD_ANNUAL_RECURRING (decimal(18,2)): Annual recurring revenue amount in USD.
25. USD_ADJUSTED_ANNUAL_RECURRING (decimal(18,2)): Adjusted annual recurring revenue amount in USD.


prod_corp_snowflake_provisioning_shared.salesforce.request:
* REQUEST_ID (string): The unique identifier for each request.
* CREATED_DATE (timestamp): The date and time when the request was initially created.
* LAST_MODIFIED_DATE (timestamp): The date and time when the request was last modified.
* CREATED_BY_EMPLOYEE_ID (string): The employee ID of the person who created the request.
* LAST_MODIFIED_BY_EMPLOYEE_ID (string): The employee ID of the person who last modified the request.
* ACCOUNT_ID (string): The account ID associated with the request.
* ASSIGNED_TO_EMPLOYEE_ID (string): The employee ID of the person to whom the request is assigned.
* RECORD_TYPE_ID (string): The ID of the record type for the request.
* VENDOR_CONTACT_ID (string): The ID of the vendor contact associated with the request.
* CUSTOMER_RELIEF_OWNER_EMPLOYEE_ID (string): The employee ID of the customer relief owner linked to the request.
* SALES_REPRESENTATIVE_EMPLOYEE_ID (string): The employee ID of the sales representative responsible for the request.
* SALES_DIRECTOR_EMPLOYEE_ID (string): The employee ID of the sales director overseeing the request.
* CREDIT_REQUEST_REASON_CODE (string): The code indicating the reason for requesting a credit.
* CREDIT_TYPE (string): The type of credit requested.
* CURRENCY_CODE (string): The currency code used in the request.
* REQUEST_TYPE (string): The type of request being made.
* REQUEST_STATUS (string): The current status of the request.
* CUSTOMER_RELIEF_BUCKET (string): A category or bucket for customer relief associated with the request.
* PRODUCTS_AFFECTED (string): A list of products affected by the request.
* PAYMENT_TERMS (string): The payment terms agreed upon for the request.
* SERVICES_PURCHASED (string): The services purchased as part of the request.
* CONTRACT_TERM (string): The term of the contract associated with the request.
* RECENT_DISPUTES_ISSUES (string): Details of any recent disputes or issues related to the request.
* APPROVAL_PERIOD (string): The approval period for the request.
* PRIOR_PERIOD_INVOICE_UNPAID (string): Indicates if there are any unpaid invoices from prior periods.
* FINANCE_NAV_CUSTOMER_CODE (string): The finance NAV customer code related to the request.
* FINANCE_CUSTOMER_CODE_1 (string): An additional finance customer code.
* FINANCE_CUSTOMER_CODE_2 (string): Another additional finance customer code.
* FINANCE_CUSTOMER_CODE_3 (string): Yet another additional finance customer code.
* EXTENSION_TO_TERM (string): Information regarding any extension to the contract term.
* APPROVED_PRODUCTS (string): Products approved as part of the request.
* CANCELLAION_SUBMITTED (string): Indicates if a cancellation has been submitted.
* RECORD_TYPE_NAME (string): The name of the record type for the request.
* RATIONALE_FOR_APPROVAL (string): The rationale provided for approving the request.
* TRADING_PARTNERS_TEXT (string): Text describing trading partners involved in the request.
* RECENT_DISPUTES_ISSUES_COMMENTS (string): Comments on recent disputes or issues.
* CUSTOMER_ASK (string): Details of the customer’s request.
* REQUEST_EXPIRATION_DATE (date): The expiration date of the request.
* PRODUCTS_FOR_SUSPENSION_HOLD (string): Products that are on suspension or hold.
* REQUEST_NAME (string): The name or title of the request.
* TOTAL_RATE_REDUCTION (decimal(18,2)): The total rate reduction requested.
* REASON_CODE (string): Code indicating the reason for the request.
* CAMPAIGN_ID (string): The ID of the campaign associated with the request.
* OPPORTUNITY_ID (string): The ID of the sales opportunity linked to the request.
* CURRENT_MONTHLY_PRICE (decimal(18,2)): The current monthly price before any requested changes.
* PROPOSED_MONTHLY_PRICE (decimal(18,2)): The proposed new monthly price after changes.
* USD_TOTAL_RATE_REDUCTION (decimal(18,2)): The total rate reduction requested in USD.
* OPPORTUNITY_CURRENCY_KEY (decimal(10,0)): A key related to the currency for the opportunity.
* BILLING_STATUS (string): The billing status associated with the request.
* CREDIT_AMOUNT_REQUESTED (decimal(18,0)): The amount of credit requested.


prod_corp_snowflake_provisioning_shared.salesforce.subscription:
1. SUBSCRIPTION_ID: string - A unique identifier for each subscription.
2. CONTRACT_ID: string - The identifier for the contract associated with the subscription.
3. NET_PRICE: decimal(14,2) - The net price of the subscription.
4. QUANTITY: decimal(12,2) - The quantity associated with the subscription.
5. TRADING_PARTNER_ACCOUNT_ID: string - The identifier for the trading partner account.
6. PRODUCT_ID: string - The identifier for the product associated with the subscription.
7. SUBSCRIPTION_START_DATE: timestamp - The start date of the subscription period.
8. SUBSCRIPTION_END_DATE: timestamp - The end date of the subscription period.
9. SUBSCRIPTION_STATUS: string - The current status of the subscription (e.g., active, pending, cancelled).
10. ETL_CREATED_TIMESTAMP: timestamp - The timestamp when the record was created in the ETL process.
11. ETL_UPDATED_TIMESTAMP: timestamp - The timestamp when the record was last updated in the ETL process.
12. IS_ACTIVE: decimal(1,0) - A flag indicating whether the subscription is active (1) or not (0).
13. TERMINATED_DATE: timestamp - The date the subscription was terminated, if applicable.
14. CONNECTION_LOOKUP_ID: string - An identifier used for connection lookup.
15. SPS_FOR_3PL_LOCATION_ID: string - Identifier for the location used by the third-party logistics provider.
16. TOTAL_DOCUMENTS: decimal(18,0) - The total number of documents associated with the subscription.
17. OVERAGE_RATE: decimal(18,2) - The overage rate applied to the subscription.
18. BLENDED_DOC_RATE_DROP_SHIP_DOCUMENTS: decimal(14,0) - The blended document rate for drop-ship documents.
19. BLENDED_DOC_RATE_STANDARD_DOCUMENT: decimal(14,0) - The blended document rate for standard documents.



The table prod_networkdata.reporting_gold.customer_visible_transaction_volume_daily contains aggregated information about customer-visible transaction volumes over the last 90 days. Below is a detailed description of each column, including the name, type, and description:
1. dc4SenderId (string): This column contains the DC4 ID of the sender.
2. dc4SenderName (string): This column contains the sender's name as reflected in DC4.
3. senderOrgId (string): This column contains the SPS Identity Organization ID of the sender.
4. dc4ReceiverId (string): This column contains the DC4 ID of the receiver.
5. dc4ReceiverName (string): This column contains the receiver's name as reflected in DC4.
6. receiverOrgId (string): This column contains the SPS Identity Organization ID of the receiver.
7. docType (string): This column specifies the type of documents, usually EDI X12 or EDIFACT.
8. startDay (date): This column represents the date of the earliest event that contributed to the transaction.
9. totalVolume (bigint): This column contains the total number of transactions.
10. errorVolume (bigint): This column specifies the number of errored transactions.
The relationship with account ID table seems to be as follows SELECT
    txn.`dc4SenderId`,
    txn.`dc4SenderName`,
    sender_acc.`ACCOUNT_ID` AS sender_account_id,
    txn.`dc4ReceiverId`,
    txn.`dc4ReceiverName`,
    receiver_acc.`ACCOUNT_ID` AS receiver_account_id,
    txn.`docType`,
    txn.`startDay`,
    txn.`totalVolume`,
    txn.`errorVolume`
FROM
    `prod_networkdata`.`reporting_gold`.`customer_visible_transaction_volume_daily` AS txn
LEFT JOIN
    `snowflake_corp`.`sps_identity`.`account` AS sender_identity
ON
    txn.`senderOrgId` = sender_identity.`ACCOUNT_ID`
LEFT JOIN
    `snowflake_corp`.`salesforce`.`account` AS sender_acc
ON
    sender_identity.`SALESFORCE_ID` = sender_acc.`ACCOUNT_ID`
LEFT JOIN
    `snowflake_corp`.`sps_identity`.`account` AS receiver_identity
ON
    txn.`receiverOrgId` = receiver_identity.`ACCOUNT_ID`
LEFT JOIN
    `snowflake_corp`.`salesforce`.`account` AS receiver_acc
ON
    receiver_identity.`SALESFORCE_ID` = receiver_acc.`ACCOUNT_ID`


Current setup of 00 in Databricks
datasets = {
#Salesforce Account:
    "account": "prod_corp_snowflake_provisioning_shared.salesforce.account",
#Salesforce Support Cases:
    "case":"prod_corp_snowflake_provisioning_shared.salesforce.case",
    "case_history":"prod_corp_snowflake_provisioning_shared.salesforce.case_history",
#Salesforce Contract:
    "contract": "prod_corp_snowflake_provisioning_shared.salesforce.contract",
#Salesforce Onboarding Projects:
    "implementation_project":"prod_corp_snowflake_provisioning_shared.salesforce.implementation_project",
#Salesforce Opportunities
    "opportunity":"prod_corp_snowflake_provisioning_shared.salesforce.opportunity",
    "opportunity_product":"prod_corp_snowflake_provisioning_shared.salesforce.opportunity_product",
#Salesforce Requests:
    "request":"prod_corp_snowflake_provisioning_shared.salesforce.request",
#Salesforce Subscription Renewal:
    "subscription":"prod_corp_snowflake_provisioning_shared.salesforce.subscription",

#Network Usage:
    #"reporting_service_data":"corpdev_snowflake_cs_org_product_catalog.public.reporting_service_data",
    #"customer_visible_transaction_volume_daily":"prod_networkdata.reporting_gold.customer_visible_transaction_volume_daily",
    #"orderexchange_gold":"prod_networkdata.orderexchange_gold",
#Network Errors:
    #"reporting_service_dets_events": "corpdev_snowflake_cs_org_product_catalog.public.reporting_service_dets_events"
}
