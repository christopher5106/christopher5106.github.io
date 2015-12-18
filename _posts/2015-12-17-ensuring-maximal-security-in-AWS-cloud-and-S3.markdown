---
layout: post
title:  "Ensuring maximal security in the AWS cloud and S3."
date:   2015-12-13 23:00:51
categories: security
---

Here are a few advices about securing your account and data.

Keep in mind these three principles :

1. **Simplicity is the main factor of security**. If your security settings are complex to review, then it might be less secure.

2. **Work with white lists** : everything is denied by default. Except for the users in the white list. Working with black list is an anti-pattern.

3. **Seperate physical and logical world**. In particular, create one credential key per couple (application, computer), and logical groups to give these physical entities access security rights.


# Root account

Advice \#1 : **delete your root access key and add MFA authentification with your mobile**.

After that, you'll certainly feel better : from now on, it is more difficult for anyone to change security rights of your IAM users and buckets through the web interface :).

Except in case of very bad misconfigurations, which you'll naturally avoid or do with a lot of care, such as :
- allowing full administration rights to a user,
- making some data public, or
- granting access to S3 buckets to other AWS accounts.


# Strategy for managing YOU as a user, and your access keys

A very usual experience with AWS is the following one : you begin with a first key that you set in the `.aws/credentials` file or in the `.bash_profile` of the first computer you use AWS from. But long later, you end with more keys and users, some of them re-used across computers and services... not knowing any more exactly which computer is allowed to access to what, which keys, where etc. Keys dispatched in the nature...

Advice \#2 : **create as many users as you have computers**

That could be :
- *user christopher-work*
- *user christopher-home*
- *user christopher-laptop*
- *user christopher-phone*
- *user christopher-vacation*

that you will add to logical groups.

With this configuration, you can

- revoke any infested or robbed computer (in particular your [phone or laptop](https://en.wikipedia.org/wiki/Laptop_theft)!).

- if you want to be perfectly secure, add only the currently used computer in a group and have all other users excluded

- add only the most frequently used computers, and leave the less frequently used ones outside of the group, you'll add them only when necessary.

Why not using multiple user credentials instead of creating multiple users ? If you create only one user,

- you do not know which one corresponds to which computer when you want to revoke (not tags available)

- you cannot grant fine-grained rights to your different computers, such as for example, the computer at work may access your personal account, but the computer at home will not; or the contrary around. When you revoke a right, you would revoke all your computer. You cannot seperate frequently used and less frequently computers.

The *main limitations to groups in AWS* :

- you cannot grant right to groups in encryption keys

- you cannot grant rights to groups in bucket policies

In both cases, you have to grant all your users independently.

Easy to follow:

- **one user per computer and never re-use, never download a credentials.** Copy them once to your `.aws/credentials` file, to one computer only.

- **one group per logical application**

- **access policies in groups**. A group is the first white list.


# Applications

Your computer are all set for **AWS CLI**, now it's time to code your application.

Advice \#3: **avoid using a credential in an application, create a group for the work on your computer/localhost, and a role for your EC2 instances**

AWS SDK will get directly its credentials from the role for your EC2 instances, and from [your environment variables](http://docs.aws.amazon.com/aws-sdk-php/v2/guide/credentials.html#environment-credentials) for your computer.

Do not forget to use a credential cache.


# Encrypting data


Advice \#4 : **create a second level of security, with KMS encryption**.

In IAM > encryption keys panel, create an encryption key and allow a user to use it. It is an **encrypting key white list** (the white list number 2).

To ensure everything in the bucket is encrypted :

  {
  "Version":"2012-10-17",
  "Id":"PutObjPolicy",
  "Statement":[{
       "Sid":"DenyUnEncryptedObjectUploads",
       "Effect":"Deny",
       "Principal":"*",
       "NotPrincipal": {
         "AWS": "arn:aws:iam::ACCOUNT_ID:user/USER_NAME" },
       "Action":"s3:PutObject",
       "Resource":"arn:aws:s3:::MYSECUREBUCKET/*",
       "Condition":{
          "StringNotEquals":{
             "s3:x-amz-server-side-encryption":"aws:kms"
          }
       }
    }
  ]
  }


Give access to your user by attaching a policy to an application group and role :

  {
      "Version": "2012-10-17",
      "Statement": [
          {
              "Sid": "Stmt",
              "Effect": "Allow",
              "Action": [
                  "s3:*"
              ],
              "Resource": [
                  "arn:aws:s3:::MYBUCKET",
                  "arn:aws:s3:::MYBUCKET/*",
                  "arn:aws:s3:::MYBUCKET-encrypted",
                  "arn:aws:s3:::MYBUCKET-encrypted/*"
              ]
          }
      ]
  }

It is so easy to add an access to your bucket ! To avoid a misconfiguration, I add a security with a DENY statement to the bucket policy:

  {
  	"Version": "2012-10-17",
  	"Id": "PutObjPolicy",
  	"Statement": [
  		{
  			"Sid": "DenyUnEncryptedObjectUploads",
  			"Effect": "Deny",
  			"NotPrincipal": {
  				"AWS": "arn:aws:iam::ACCOUNT_ID:user/USER_NAME"
  			},
  			"Action": "s3:PutObject",
  			"Resource": "arn:aws:s3:::MYSECUREBUCKET/*"
  		},
  		{
  			"Sid": "DenyUnEncryptedObjectUploads",
  			"Effect": "Deny",
  			"Principal": {
  				"AWS": "arn:aws:iam::ACCOUNT_ID:user/USER_NAME"
  			},
  			"Action": "s3:PutObject",
  			"Resource": "arn:aws:s3:::MYSECUREBUCKET/*",
  			"Condition": {
  				"StringNotEquals": {
  					"s3:x-amz-server-side-encryption": "aws:kms"
  				}
  			}
  		}
  	]
  }

The order in which the policies are evaluated has no effect on the outcome of the evaluation. DENY have priority on ALLOW statements.

We can regret that policy does not allow a `s3:x-amz-server-side-encryption-aws-kms-key-id` condition key which would be very nice [feature request](https://forums.aws.amazon.com/thread.jspa?messageID=609709).

We have in this bucket policy a **bucket white list** (white list number 3).



# Granting / Revoking checklist

It's now a mess again. Here is the checklist :


**Adding a computer** :

- create a user

- add it to a group

- add it to a bucket white list

- add it to an encryption key white list

Remember : you cannot add a group to a bucket policy or an encryption list.

When revoking, do the contrary.

**Adding an application** :

- create a group (for your computer and physical user to which you will share your development code)

- create a role (for your EC2 instance)

- create an encryption key
