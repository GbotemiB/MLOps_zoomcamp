from prefect_email import EmailServerCredentials

credentials = EmailServerCredentials(
    username="<email placeholder>",
    password="<password placeholder>",  # must be an app password
)
credentials.save("gmail-block", overwrite=True)
