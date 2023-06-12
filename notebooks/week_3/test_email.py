from prefect import flow
from prefect_email import EmailServerCredentials, email_send_message

@flow()
def example_email_send_message_flow(email_address):
    email_credentials_block = EmailServerCredentials.load("gmail-block")
    subject = email_send_message.with_options(name=f"email {email_address}").submit(
        email_server_credentials=email_credentials_block,
        subject="Example Flow Notification using Gmail",
        msg="This proves email_send_message works!",
        email_to=email_address,
    )

example_email_send_message_flow("<email placeholder>")
