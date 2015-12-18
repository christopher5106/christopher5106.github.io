---
layout: post
title:  "Ensuring maximal security in the AWS cloud and S3."
date:   2015-12-13 23:00:51
categories: security
---

Here are a few advices about securing your account and data.

Keep in mind these three principles :

1. **Simplicity is the main factor of security**. If your security settings are complex to review, then it might be less secure.

2. **Work with white lists** : default configuration should be everything denied. Then add authorization to the allowed users : this is the white list. Working with black list is an anti-pattern.

3. **Separate physical and logical logics**. In particular, create one credential key per computer, and logical groups to give these physical entities access security rights.


# Root account

**Advice \#1 : delete your root access key and add MFA authentification with your mobile**.

After that, you'll certainly feel better : from now on, it is more difficult for anyone to change security rights of your IAM users and buckets through the web interface :).

Since there isn't any root access key any more, this could happen only in case of very bad misconfigurations, which you'll naturally avoid or do with a lot of care, such as :

- allowing full administration rights to a user,

- making some data public, or

- granting access to S3 buckets to other AWS accounts.


# Strategy for managing YOU as a user, and your access keys

A very usual experience with AWS is the following one : you begin with a first key that you set in the `.aws/credentials` file or in the `.bash_profile` of the first computer you use AWS from. But long later, you end up with more keys and more users, some of them re-used across computers and services... not knowing any more exactly which computer is allowed to access to what, which keys, where etc. Keys dispatched in the nature...

**Advice \#2 : create as many users as you have computers**

That could be :

- *user christopher-work*

- *user christopher-home*

- *user christopher-laptop*

- *user christopher-phone*

- *user christopher-vacation*

that you will add to logical groups.

With this configuration, you can

- revoke any infested or robbed computer (in particular your [phone or laptop](https://en.wikipedia.org/wiki/Laptop_theft)!).

- if you want to be perfectly secure, add only the currently used computer in a group and have all other computers outside

- add only the most frequently used computers, and leave the less frequently used ones outside of the group, you'll add them only when necessary.

Why not using multiple user credentials instead of creating multiple users ? If you create only one user :

- you do not know which credential key corresponds to which computer in the case of revocation need (not tags available)

- you cannot grant fine-grained rights to your different computers, such as for example, the computer at work may access your personal account, but the computer at home will not access your work account; or the contrary around. When you revoke a right, you would revoke all your computers. You cannot separate frequently used and less frequently computers.

The *main limitations to groups in AWS* :

- you cannot grant rights to groups in encryption keys

- you cannot grant rights to groups in bucket policies

In both cases, you have to grant all your users independently.

**Never re-use, never download a credentials.** Copy them once to your `.aws/credentials` file, to one computer only.





# Applications

Your computer are all set for **AWS CLI**, now it's time to code and run your application.

**Advice \#3: create one group and one role per application and attach them access policies.**

AWS SDK will get directly its credentials from the role for your EC2 instances, and from [your environment variables](http://docs.aws.amazon.com/aws-sdk-php/v2/guide/credentials.html#environment-credentials) for your computer :

- the group is for the development work of your user computers, to run the code on the localhost, access data...

- the role is for the EC2 instances to run the preproduction and production environments. Do not forget to use a credential cache.

Groups/roles act as **group/role white list**, listing allowed user computers and instances.

**Avoid using any credential in an application**


# Encrypting data


**Advice \#4 : encrypt your data**.

KMS encryption is a second level of security.

In IAM > encryption keys panel, create an encryption key and allow a user to use it. It is the **encrypting key white list** that acts as a second level of security.

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

We can regret that policy grammar does not allow a `s3:x-amz-server-side-encryption-aws-kms-key-id` condition key which would be very nice [feature request](https://forums.aws.amazon.com/thread.jspa?messageID=609709).


To give access to a computer or application, simply attach a policy to a group/role white list,

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

It is so easy to add an access to your bucket ! To avoid a misconfiguration, you can add a third security with a DENY statement to the bucket policy:

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
    			"Action": "s3:*",
    			"Resource": "arn:aws:s3:::MYSECUREBUCKET-encrypted/*"
    		},
    		{
    			"Sid": "DenyUnEncryptedObjectUploads",
    			"Effect": "Deny",
    			"Principal": {
    				"AWS": "arn:aws:iam::ACCOUNT_ID:user/USER_NAME"
    			},
    			"Action": "s3:PutObject",
    			"Resource": "arn:aws:s3:::MYSECUREBUCKET-encrypted/*",
    			"Condition": {
    				"StringNotEquals": {
    					"s3:x-amz-server-side-encryption": "aws:kms"
    				}
    			}
    		}
    	]
    }

The order in which the policies are evaluated has no effect on the outcome of the evaluation. DENY have priority on ALLOW statements. We have with this bucket policy a third white list, **bucket white list**.



# Granting / Revoking checklist

It's now a mess again. Here is the checklist :


**Adding a computer** :

- create a user

- add it to a group (white list)

- add it to a bucket policy (white list)

- add it to an encryption key (white list)

Remember : you cannot add a group to a bucket policy or an encryption list.

When revoking, do the contrary.

**Adding an application** :

- create a group (for your computers and other physical users with which you will share your development)

- create a role (for your EC2 instance)

- create an encryption key
